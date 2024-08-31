import torch
import whisper
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from subprocess import CalledProcessError, run
from typing import List, Callable, Optional
from collections import defaultdict
import pathlib
from tqdm import tqdm
from whisper_interpretability.global_utils import device, BaseActivationModule
from whisper_interpretability import whisper_repo as wr
from jaxtyping import Float
from torch import Tensor
import csv
from sklearn.decomposition import PCA

from enum import Enum

class DataType(Enum):
    LOVE = "love"
    ANGER = "anger"

class AnalysisType(Enum):
    PCA = "pca"
    GAUSS = "gaussian"
    SP = "sparse"
    ENC = "encoded"



# ... (keep your existing imports and functions) ...


class WhisperActivationCache(BaseActivationModule):
    """
    Use hooks in BaseActivationModule to cache intermediate activations while running forward pass
    """

    def __init__(
        self,
        hook_fn: Optional[Callable] = None,
        model: Optional[torch.nn.Module] = None,  # if model not provided load whisper model by name
        model_name: str = "tiny",
        activations_to_cache: list = ["encoder.blocks.0"],  # pass "all" to cache all activations
    ):
        self.model = (
            model.to(device) if model is not None else whisper.load_model(model_name).to(device)
        )
        self.activations_to_cache = activations_to_cache
        self.named_modules = list({name: mod for name, mod in self.model.named_modules()})
        super().__init__(
            model=self.model,
            activations_to_cache=self.activations_to_cache,
            hook_fn=hook_fn,
        )

    def custom_forward(
        self, model: torch.nn.Module, mels: Float[Tensor, "bsz seq_len n_mels"]
    ):  # noqa: F821
        options = whisper.DecodingOptions(without_timestamps=False, fp16=(device == "cuda"))
        output = model.decode(mels, options)
        return output

    def _get_caching_hook(self, name):
        # custom caching function for whisper
        def hook(module, input, output):
            if "decoder" in name:
                # we don't cache the first activations that correspond to the sos/lang tokens
                if output.shape[1] > 1:
                    del self.activations[f"{name}"]
                    return
            output_ = output.detach().cpu()
            if name in self.activations:
                self.activations[f"{name}"] = torch.cat(
                    (self.activations[f"{name}"], output_), dim=1
                )
            else:
                self.activations[f"{name}"] = output_

        return hook

def get_backward_attn_hook(name, attn_grads):
    def hook_fn(module, input_, output_):
        attn_grads[name] = input_[0]

    return hook_fn


def get_forward_attn_hook(name, attn_scores):
    def hook_fn(module, input_, output_):
        attn_scores[name] = input_[0].detach()

    return hook_fn


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def get_encoder_attn_map(attn_scores, attn_grads):
    R = None
    for layer in attn_scores.keys():
        attn_score = attn_scores[layer]
        attn_grad = attn_grads[layer]
        cam = avg_heads(attn_score, attn_grad)
        if R is None:
            R = torch.eye(attn_score.shape[-1], attn_score.shape[-1])
        R += apply_self_attention_rules(R, cam)

    return R


def trim_audio(
    array: np.array,
    start_time: float,
    end_time: float,
    sample_rate: int = 16_000,
):
    """
    Trim the audio file base array to n_samples, as expected by the encoder.
    """
    start_frame = int(sample_rate * start_time)
    end_frame = int(sample_rate * end_time)

    return array[start_frame:end_frame]


def load_audio(file: str, sample_rate_hz: int = 16_000):
    """
    Taken from Whisper repo: https://github.com/openai/whisper/blob/main/whisper/audio.py

    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sample_rate_hz: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate_hz),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


tokenizer = wr.tokenizer.get_tokenizer(multilingual=True)
WHISPER_MODEL: str = "tiny"


base_whisper_model = whisper.load_model(WHISPER_MODEL)
ModelDims = base_whisper_model.dims
model = wr.model.Whisper(ModelDims)  # use our own model definition with attn hook points
model.load_state_dict(base_whisper_model.state_dict())

# audio_path = "output.wav"
# audio = load_audio(audio_path)
# audio = trim_audio(audio, start_time=0, end_time=10)
# audio = whisper.pad_or_trim(audio.flatten())
# mels = torch.tensor(whisper.log_mel_spectrogram(audio)).to(device)

def attention_grid():
    attn_grads = {}
    for name, mod in model.named_modules():
        if "encoder" in name and "attn_hook" in name:
            mod.register_full_backward_hook(get_backward_attn_hook(name, attn_grads))

    attn_scores = {}
    for name, mod in model.named_modules():
        if "encoder" in name and "attn_hook" in name:
            mod.register_forward_hook(get_forward_attn_hook(name, attn_scores))

    model.to(device)
    mels = mels.to(device)
    output = model.embed_audio(mels.unsqueeze(0))
    output.backward(torch.ones_like(output))

    R = get_encoder_attn_map(attn_scores, attn_grads)

    attention_grid = R.detach().cpu().numpy()
    # print(R.shape)
    plt.figure(figsize=(10, 10))  # Set figure size
    plt.imshow(attention_grid, cmap='viridis', aspect='auto')  # Plot heatmap

    # Add a colorbar to show the scale
    plt.colorbar()

    # Add labels and title (optional)
    plt.title('Attention Grid Visualization')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    # Save the plot as an image file
    plt.savefig('attention_grid.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

# ... existing code ...


def get_activations(
    audio_dir: str,
    activations_to_cache: list = [
        "encoder.blocks.0",
        "encoder.blocks.1",
        "encoder.blocks.2",
        "encoder.blocks.3",
    ],
    model_name: str = "tiny",
    batch_size: int = 50,
    out_dir: str = "./out/whisper_activations",
    class_label: str = "en_big",
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")

    actv_cache = WhisperActivationCache(
        model_name=model_name, activations_to_cache=activations_to_cache
    )

    all_actvs = defaultdict(list)  # layer_name: activations

    audio_paths = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    for i in tqdm(range(0, len(audio_paths), batch_size)):
        batch_paths = audio_paths[i:i+batch_size]
        batch_audio = []

        for audio_path in batch_paths:
            audio = load_audio(audio_path)
            audio = trim_audio(audio, start_time=0, end_time=10)
            audio = whisper.pad_or_trim(audio.flatten())
            mels = whisper.log_mel_spectrogram(audio)
            batch_audio.append(mels)

        batch_mels = torch.stack(batch_audio).to(device)

        actv_cache.reset_state()
        actv_cache.forward(batch_mels)

        for layer in activations_to_cache:
            actvs = actv_cache.activations[f"{layer}"].to(dtype=torch.float32).to("cpu")
            all_actvs[layer].append(actvs)

            # Save each activation to disk
            temp_dir = pathlib.Path(out_dir) / "en_save_activations"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"{model_name}_{layer}_{class_label}_batch_{i}.pt"
            torch.save(actvs, temp_file)
            print(f"Saved temporary activation for layer {layer}, batch {i} to {temp_file}")

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for layer, actvs in all_actvs.items():
        actvs = torch.cat(actvs, dim=0).mean(dim=0)  # mean over all batches but retain sequence length
        torch.save(actvs, f"{out_dir}/{model_name}_{layer}_{class_label}")
        print(f"Saved {actvs.shape} activations for layer {layer} to disk")

def process_saved_activations(
    temp_dir: str = "./out/whisper_activations/en_save_activations",
    out_dir: str = "./out/whisper_activations",
    model_name: str = "tiny",
    class_label: str = "en_big",
    activations_to_cache: list = [
        "encoder.blocks.0",
        "encoder.blocks.1",
        "encoder.blocks.2",
        "encoder.blocks.3",
    ],
):
    temp_dir = pathlib.Path(temp_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in activations_to_cache:
        all_actvs = []
        layer_files = list(temp_dir.glob(f"{model_name}_{layer}_{class_label}_batch_*.pt"))

        for file in tqdm(layer_files, desc=f"Processing {layer}"):
            actvs = torch.load(file)
            all_actvs.append(actvs)

        if all_actvs:
            combined_actvs = torch.cat(all_actvs, dim=0)
            mean_actvs = combined_actvs.mean(dim=0)  # mean over all batches but retain sequence length

            out_file = out_dir / f"{model_name}_{layer}_{class_label}"
            torch.save(mean_actvs, out_file)
            print(f"Saved {mean_actvs.shape} mean activations for layer {layer} to {out_file}")
        else:
            print(f"No activations found for layer {layer}")

    print("Finished processing all saved activations.")

import torch
from sklearn.decomposition import PCA

def reshape_tensor(tensor, target_shape):
    assert len(target_shape) == 3, "Target shape must be 3-dimensional (batch, seq, features)"
    assert target_shape[0] == 1, "Batch size must be 1"

    original_seq, original_features = tensor.shape
    target_seq, target_features = target_shape[1], target_shape[2]

    # Step 2: Pad or truncate along the sequence dimension
    if original_seq < target_seq:
        # Pad
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, target_seq - original_seq))
    else:
        # Truncate
        padded = tensor[:target_seq]

    # Step 3: Add a new dimension at the beginning for batch
    result = padded.unsqueeze(0)

    return result

# Example usage:
class LanguageSteeringModule:
    def __init__(
        self,
        activations_to_steer: list,
        model_name: str = "tiny",
        model: torch.nn.Module = None,
        steering_factor: float = 1.0,
        steering_vector: torch.Tensor = None,  # Single steering vector
        audio_path: str = "",
        data_type: DataType = DataType.ANGER,
        analysis_type: AnalysisType = AnalysisType.PCA,
    ):
        self.model_name = model_name
        if model is None:
            self.model = whisper.load_model(model_name)
        else:
            self.model = model
        self.model = self.model.to(device)
        self.activations_to_steer = activations_to_steer
        self.hooks = []
        self.steering_factor = steering_factor
        self.steering_vector = steering_vector
        self.audio_path = audio_path
        self.data_type = data_type
        self.analysis_type = analysis_type

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.activations_to_steer:
                self.hooks.append(
                    module.register_forward_hook(
                        self.get_lang_steering_hook(
                            self.steering_vector,
                            self.steering_factor,
                            name,
                            self.audio_path,
                            self.data_type,
                            self.analysis_type
                        )
                    )
                )

    def forward(self, x):
        self.register_hooks()
        options = whisper.DecodingOptions(without_timestamps=False, fp16=(device == "cuda"))
        output = self.model.decode(x, options)
        self.remove_hooks()
        return output

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    @staticmethod
    def get_lang_steering_hook(steering_vector, steering_factor, name, audio_path, data_type, analysis_type):
        def hook(module, input, output):
            output_ = output.detach()
            new_output = output_ + (steering_factor * steering_vector.half().to(device))
            # Measure the change between output_ and new_output
            change = torch.norm(new_output - output_)
            output_[:] = new_output

            # saves the steering vector
            torch.save(new_output, f'results/cross_steering/{audio_path.split("/")[-1].split(".")[0]}_{data_type.value}_{analysis_type.value}_{name.split(".")[-1]}_{steering_factor}.pt')

            return output_
        return hook

# ... rest of the existing code ...

def apply_pca_and_reshape(input_tensor, output_shape, device):
    """
    Perform PCA on the input tensor and reshape it to match the output tensor's shape.

    Parameters:
        input_tensor (torch.Tensor): The input tensor to transform.
        output_shape (tuple): The desired shape for the output tensor.
        device (torch.device): The device to run the operations on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: The transformed tensor with the same shape as the output tensor.
    """
    # Detach the input tensor to avoid affecting gradients
    input_tensor_ = input_tensor.detach()

    # Reshape the input tensor to 2D for PCA
    input_tensor_reshaped = input_tensor_.reshape(-1, input_tensor_.shape[-1]).cpu().numpy()

    # Perform PCA to match the last dimension of the output tensor
    pca = PCA(n_components=output_shape[-1])
    input_tensor_pca = pca.fit_transform(input_tensor_reshaped)

    # Convert back to tensor and reshape to match the output dimensions
    input_tensor_pca = torch.tensor(input_tensor_pca, device=device).reshape(-1, output_shape[-1])

    # Broadcast the PCA-transformed tensor to match the output shape
    output_tensor = reshape_tensor(input_tensor_pca, output_shape)

    return output_tensor


def steer_audio(audio_path: str, steering_module: LanguageSteeringModule):
    # Load and preprocess audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    # Use the steering module to process the input
    result = steering_module.forward(mel)

    return result.text

def process_tensor(tensor_path):
    array = torch.load(tensor_path)
    tensor = torch.tensor(array)
    reshaped = reshape_tensor(tensor=tensor, target_shape=(1, 1500, 384))

    return reshaped
#* Steering Factors list

def get_pca_vector(tensor, target_shape):
    # Reshape the input tensor to 2D for PCA
    reshaped_tensor = tensor.reshape(-1, tensor.shape[-1])  # Shape: [425, 1600]

    # Apply PCA to reduce the last dimension from 1600 to 384
    pca = PCA(n_components=384)
    tensor_pca = pca.fit_transform(reshaped_tensor)  # Shape: [425, 384]

    # Convert back to tensor
    tensor_pca = torch.tensor(tensor_pca, device=tensor.device)

    # Pad or truncate to match the target shape
    total_elements = torch.prod(torch.tensor(target_shape))
    if tensor_pca.numel() < total_elements:
        # Pad
        padding = torch.zeros(total_elements - tensor_pca.numel(), device=tensor.device)
        tensor_pca = torch.cat([tensor_pca.reshape(-1), padding])
    elif tensor_pca.numel() > total_elements:
        # Truncate
        tensor_pca = tensor_pca.reshape(-1)[:total_elements]

    # Reshape to match the target shape
    output_tensor = tensor_pca.reshape(target_shape)  # Shape: [1, 1500, 384]

    return output_tensor

def get_projection_encode_vector(path, shape=(1, 1500, 384)):
    array = torch.load(path)

    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        tensor = array.clone().detach()
    else:
        raise TypeError(f"Unexpected type: {type(array)}. Expected numpy.ndarray or torch.Tensor.")

    if len(tensor.shape) > 2:
        tensor = tensor[0]
    return reshape_tensor(tensor, shape)

left = np.arange(-6, -1, 0.1)
middle = np.arange(-1, 1.01, 0.01)  # 1.01 to include 1
right = np.arange(1.1, 6.1, 0.1)  # 6.1 to include 6

steering_factors_cumulative = np.concatenate((left, middle, right))

#* load in vectors (all xl)
shape = (1, 1500, 384)
# #* love vectors
pca_love = get_pca_vector(torch.load("vectors/gpt2-xl/data/love-pca.pt", map_location=torch.device(device)), shape)
encoded_love = get_projection_encode_vector('vectors/gpt2-xl/love-encoded.pt')
gaussian_love = get_projection_encode_vector('vectors/gpt2-xl/love_rp_gaussian.pt')
sparse_love = get_projection_encode_vector('vectors/gpt2-xl/love_rp_sparse.pt')
print("Shape of pca_love:", pca_love.shape)
print("Shape of encoded_love:", encoded_love.shape)
print("Shape of gaussian_love:", gaussian_love.shape)
print("Shape of sparse_love:", sparse_love.shape)

# #* anger vectors
pca_anger = get_pca_vector(torch.load("vectors/gpt2-xl/data/anger-pca.pt", map_location=torch.device(device)), shape)
encoded_anger = get_projection_encode_vector('vectors/gpt2-xl/anger-encoded.pt')
gaussian_anger = get_projection_encode_vector('vectors/gpt2-xl/anger_rp_gaussian.pt')
sparse_anger = get_projection_encode_vector('vectors/gpt2-xl/anger_rp_sparse.pt')
print("Shape of pca_anger:", pca_anger.shape)
print("Shape of encoded_anger:", encoded_anger.shape)
print("Shape of gaussian_anger:", gaussian_anger.shape)
print("Shape of sparse_anger:", sparse_anger.shape)


def loop_range(steering_vector, audio_path, steering_factors, analysis_type, data_type):
    results = []
    for factor in tqdm(steering_factors):
        steering_module = LanguageSteeringModule(
            activations_to_steer=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2", "encoder.blocks.3"],
            model_name="tiny",
            steering_factor=factor,
            steering_vector=steering_vector,
            audio_path=audio_path,
            data_type=data_type,
            analysis_type=analysis_type
        )

        steered_text = steer_audio(audio_path, steering_module)
        results.append((factor, steered_text))

    # Create directory for results
    # result_dir = f"results/{analysis_type.value}/{data_type.value}"
    # os.makedirs(result_dir, exist_ok=True)
    # # Save results to a CSV file
    # filename = f"{result_dir}/{audio_path.split('/')[-1].split('.')[0]}.csv"
    # with open(filename, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Steering Factor", "Steered Text"])
    #     writer.writerows(results)

    return results

def load_steering_vectors(device, shape):
    vectors = {}

    for data_type in DataType:
        vectors[data_type.value] = {
            'pca': get_pca_vector(torch.load(f"vectors/gpt2-xl/data/{data_type.value}-pca.pt", map_location=torch.device(device)), shape),
            'encoded': get_projection_encode_vector(f'vectors/gpt2-xl/{data_type.value}-encoded.pt'),
            'gaussian': get_projection_encode_vector(f'vectors/gpt2-xl/{data_type.value}_rp_gaussian.pt'),
            'sparse': get_projection_encode_vector(f'vectors/gpt2-xl/{data_type.value}_rp_sparse.pt')
        }
        # Save vectors to a directory
        # vector_dir = f"steering_vectors/{data_type.value}"
        # os.makedirs(vector_dir, exist_ok=True)
        # for analysis_type, vector in vectors[data_type.value].items():
        #     vector_path = f"{vector_dir}/{data_type.value}-{analysis_type}.pt"
        #     torch.save(vector, vector_path)

    return vectors

def get_steering_vector(vectors, data_type, analysis_type):
    return vectors[data_type.value][analysis_type.value]

def process_audio_files(base_path, steering_vectors, steering_factors):
    for data_type in DataType:
        audio_folder = os.path.join(base_path, f"cross-steering/{data_type.value}/processed")

        if not os.path.exists(audio_folder):
            print(f"Folder not found: {audio_folder}")
            continue

        wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

        for wav_file in tqdm(wav_files, desc=f"Processing {data_type.value} files"):
            audio_path = os.path.join(audio_folder, wav_file)

            for analysis_type in tqdm(AnalysisType, desc=f"Analysis types for {wav_file}", leave=False):

                # result_dir = f"results/{analysis_type.value}/{data_type.value}"
                # os.makedirs(result_dir, exist_ok=True)

                # Create the output filename
                # output_filename = f"{result_dir}/{wav_file.split('.')[0]}.csv"

                # if os.path.exists(output_filename):
                #     print(f"Skipping {analysis_type.value} for {wav_file} ({data_type.value}) - file already exists")
                #     continue

                steering_vector = get_steering_vector(steering_vectors, data_type, analysis_type)

                loop_range(
                    steering_vector=steering_vector,
                    audio_path=audio_path,
                    steering_factors=steering_factors,
                    analysis_type=analysis_type,
                    data_type=data_type
                )

base_path = '/Users/ayushjain/Development/Interp/audio-steering/whisper_interpretability/audio'
steering_vectors = load_steering_vectors(device, shape)
process_audio_files(base_path, steering_vectors, [-6,6,-0.03,0.03,0.07,-0.07])

# loop_range(sparse_anger, 'audio/cross-steering/know_this.wav', steering_factors_cumulative, AnalysisType.PCA, DataType.BASE)
#* old loop code
# def loop_range(steering_vector, steering_factors=np.arange(-6, 6, 0.01), type='pca'):
#     results = []
#     for factor in tqdm(steering_factors):
#         steering_module = LanguageSteeringModule(
#             activations_to_steer=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2", "encoder.blocks.3"],
#             model_name="tiny",
#             steering_factor=factor,
#             steering_vector=steering_vector
#         )

#         english_audio_path = "en_test.wav"
#         steered_text = steer_audio(english_audio_path, steering_module)
#         results.append((factor, steered_text))
#         print(f"Steering factor: {factor}, Steered text: {steered_text}")

#     step = steering_factors[1]-steering_factors[0]
# # Save results to a CSV file
#     with open(f"{type}_xl(anger):{steering_factors[0]}-{steering_factors[-1]}:{step}.csv", "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Steering Factor", "Steered Text"])
#         writer.writerows(results)


#* Usage with PCA
# batched_vector = torch.load("vectors/gpt2-xl/anger-pca.pt", map_location=torch.device(device))
# print(batched_vector.shape)
# steering_vector = reshape_tensor_with_pca(batched_vector, (1, 1500, 384))
# print(steering_vector.shape)

#* Usage with gaussian
# steering_vector = torch.load('vectors/love-rp-gaussian_gpt2-small.pt')
# loop_range(steering_vector=steering_vector, type='pca')
# steering_factors = np.arange(14235.999, 14236.01, 0.001)
# results = []

# from tqdm import tqdm

# for factor in tqdm(steering_factors):
#     steering_module = LanguageSteeringModule(
#         activations_to_steer=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2", "encoder.blocks.3"],
#         model_name="tiny",
#         steering_factor=factor,
#         steering_vector=steering_vector
#     )

#     english_audio_path = "en_test.wav"
#     steered_text = steer_audio(english_audio_path, steering_module)
#     results.append((factor, steered_text))
#     print(f"Steering factor: {factor}, Steered text: {steered_text}")

# # Save results to a CSV file
# with open("steering_results_14235.999999-14236.01.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Steering Factor", "Steered Text"])
#     writer.writerows(results)

# steering_module = LanguageSteeringModule(
#     activations_to_steer=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2", "encoder.blocks.3"],
#     model_name="tiny",
#     steering_factor=0,
#     steering_vector=get_pca_vector(torch.load(f"vectors/gpt2-xl/data/anger-pca.pt", map_location=torch.device(device)), shape),
# )

# english_audio_path = "/Users/ayushjain/Development/Interp/audio-steering/whisper_interpretability/audio/cross-steering/anger/processed/You were great.wav"
# steered_text = steer_audio(english_audio_path, steering_module)
# print(f"Steered text: {steered_text}")
# print(f"Steering factor: {steering_module.steering_factor}")


# class LanguageSteeringModule:
#     def __init__(
#         self,
#         activations_to_steer: list,
#         model_name: str = "tiny",
#         model: torch.nn.Module = None,
#         steering_factor: float = 1.0,
#         class_labels: list = ["es" "en_small"],  # First label is the target language, second is the source language
#     ):
#         self.model_name = model_name
#         self.class_labels = class_labels
#         if model is None:
#             self.model = whisper.load_model(model_name)
#         else:
#             self.model = model
#         self.model = self.model.to(device)
#         self.activations_to_steer = activations_to_steer
#         self.activation_dir = "out/"
#         self.hooks = []
#         self.steering_factor = steering_factor
#         self.get_steering_vectors()

#     def get_steering_vectors(self):
#         self.steering_vectors = defaultdict(list)
#         for name in self.activations_to_steer:
#             for label in self.class_labels:
#                 activations = torch.load(f"{self.activation_dir}/{self.model_name}_{name}_{label}")
#                 self.steering_vectors[name].append(activations)

#     def register_hooks(self):
#         for name, module in self.model.named_modules():
#             if name in self.activations_to_steer:
#                 self.hooks.append(
#                     module.register_forward_hook(
#                         self.get_lang_steering_hook(
#                             self.steering_vectors[name][0],
#                             self.steering_vectors[name][1],
#                             self.steering_factor,
#                         )
#                     )
#                 )

#     def forward(self, x):
#         self.register_hooks()
#         options = whisper.DecodingOptions(without_timestamps=False, fp16=(device == "cuda"))
#         output = self.model.decode(x, options)
#         self.remove_hooks()
#         return output

#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()

#     @staticmethod
#     def get_lang_steering_hook(steering_cls_vector, true_cls_vector, steering_factor):
#         def hook(module, input, output):
#             output_ = output.detach()
#             print(steering_cls_vector.shape)
#             output_[:, :, :] = (
#                 output_[:, :, :]
#                 + (steering_factor * steering_cls_vector.half().to(device))
#                 - (steering_factor * true_cls_vector.half().to(device))
#             )
#             return output_
#         return hook
# # ... rest of the existing code ...
# #### Next step is to figure out how to do steering with their library. Maybe more complicated than expected. ---

# def steer_audio(audio_path: str, steering_module: LanguageSteeringModule):
#     # Load and preprocess audio
#     audio = whisper.load_audio(audio_path)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(device)

#     # Use the steering module to process the input
#     result = steering_module.forward(mel)

#     return result.text


# steering_module = LanguageSteeringModule(
#     activations_to_steer=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2", "encoder.blocks.3"],
#     model_name="tiny",
#     steering_factor=0,
#     class_labels=["en", "fr"]  # First label is the target language, second is the source language
# )

# english_audio_path = "en_test.wav"
# steered_text = steer_audio(english_audio_path, steering_module)
# print("Steered text:", steered_text)

# print(model)

# process_saved_activations(temp_dir="out/whisper_activations/es_save_activations", out_dir="out/whisper_activations", model_name="tiny", class_label="es_big", activations_to_cache=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2"])

# process_saved_activations(temp_dir="out/whisper_activations/en_save_activations", out_dir="out/whisper_activations", model_name="tiny", class_label="en_big", activations_to_cache=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2"])