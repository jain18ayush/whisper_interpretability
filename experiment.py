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



class LanguageSteeringModule:
    def __init__(
        self,
        activations_to_steer: list,
        model_name: str = "tiny",
        model: torch.nn.Module = None,
        steering_factor: float = 1.0,
        class_labels: list = ["es" "en_small"],  # First label is the target language, second is the source language
    ):
        self.model_name = model_name
        self.class_labels = class_labels
        if model is None:
            self.model = whisper.load_model(model_name)
        else:
            self.model = model
        self.model = self.model.to(device)
        self.activations_to_steer = activations_to_steer
        self.activation_dir = "out/"
        self.hooks = []
        self.steering_factor = steering_factor
        self.get_steering_vectors()

    def get_steering_vectors(self):
        self.steering_vectors = defaultdict(list)
        for name in self.activations_to_steer:
            for label in self.class_labels:
                activations = torch.load(f"{self.activation_dir}/{self.model_name}_{name}_{label}")
                self.steering_vectors[name].append(activations)

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.activations_to_steer:
                self.hooks.append(
                    module.register_forward_hook(
                        self.get_lang_steering_hook(
                            self.steering_vectors[name][0],
                            self.steering_vectors[name][1],
                            self.steering_factor,
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
    def get_lang_steering_hook(steering_cls_vector, true_cls_vector, steering_factor):
        def hook(module, input, output):
            output_ = output.detach()
            output_[:, :, :] = (
                output_[:, :, :]
                + (steering_factor * steering_cls_vector.half().to(device))
                - (steering_factor * true_cls_vector.half().to(device))
            )
            return output_
        return hook
# ... rest of the existing code ...
#### Next step is to figure out how to do steering with their library. Maybe more complicated than expected. ---

def steer_audio(audio_path: str, steering_module: LanguageSteeringModule):
    # Load and preprocess audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    # Use the steering module to process the input
    result = steering_module.forward(mel)

    return result.text


steering_module = LanguageSteeringModule(
    activations_to_steer=["encoder.blocks.0", "encoder.blocks.1"],
    model_name="tiny",
    steering_factor=2.56,
    class_labels=["fr", "en"]  # First label is the target language, second is the source language
)

english_audio_path = "data/english_simple_wav/en1.wav"
steered_text = steer_audio(english_audio_path, steering_module)
print("Steered text:", steered_text)

# process_saved_activations(temp_dir="out/whisper_activations/es_save_activations", out_dir="out/whisper_activations", model_name="tiny", class_label="es_big", activations_to_cache=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2"])

# process_saved_activations(temp_dir="out/whisper_activations/en_save_activations", out_dir="out/whisper_activations", model_name="tiny", class_label="en_big", activations_to_cache=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2"])