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


import torch
from sklearn.decomposition import PCA
# ... rest of the existing code ...

class LanguageSteeringModule:
    def __init__(
        self,
        activations_to_steer: list,
        model_name: str = "tiny",
        model: torch.nn.Module = None,
        steering_factor: float = 1.0,
        class_labels: list = ["es", "en_small"],  # First label is the target language, second is the source language
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

    def register_hooks(self, audio_name):
        for name, module in self.model.named_modules():
            if name in self.activations_to_steer:
                self.hooks.append(
                    module.register_forward_hook(
                        self.get_lang_steering_hook(
                            self.steering_vectors[name][0],
                            self.steering_vectors[name][1],
                            self.steering_factor,
                            name,
                            audio_name,
                        )
                    )
                )

    def forward(self, x, audio_name):
        self.register_hooks(audio_name)
        options = whisper.DecodingOptions(without_timestamps=False, fp16=(device == "cuda"))
        output = self.model.decode(x, options)
        self.remove_hooks()
        return output

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    @staticmethod
    def get_lang_steering_hook(steering_cls_vector, true_cls_vector, steering_factor, layer_name, audio_name):
        def hook(module, input, output):
            output_ = output.detach()
            output_[:, :, :] = (
                output_[:, :, :]
                + (steering_factor * steering_cls_vector.half().to(device))
                - (steering_factor * true_cls_vector.half().to(device))
            )

            # Save activations
            activation_path = f"results/activations/{audio_name}_{layer_name}_sf{steering_factor}.pt"
            os.makedirs(os.path.dirname(activation_path), exist_ok=True)
            torch.save(output_, activation_path)

            return output_
        return hook

    def set_steering_factor(self, new_steering_factor):
        self.steering_factor = new_steering_factor

# Function to process multiple audio files
def process_audio_files(audio_paths: list, steering_module: LanguageSteeringModule):
    for audio_path in audio_paths:
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        for sf in [0, 1]:  # Save activations for steering factors 0 and 1
            steering_module.set_steering_factor(sf)
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(device)
            result = steering_module.forward(mel, audio_name)
            with open('results/language_results.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([audio_name, sf, result.text])

# Example usage
steering_module = LanguageSteeringModule(
    activations_to_steer=["encoder.blocks.0", "encoder.blocks.1", "encoder.blocks.2", "encoder.blocks.3"],
    model_name="tiny",
    steering_factor=1,
    class_labels=["fr", "en"]  # First label is the target language, second is the source language
)

audio_files = [
    "audio/language_steering/common_voice_en_40187662.wav",
    "audio/language_steering/common_voice_en_40187695.wav",
    "audio/language_steering/common_voice_en_40187710.wav",
    # "audio/language_steering/common_voice_fr_40191733.wav",
    # "audio/language_steering/common_voice_fr_40203015.wav",
    # "audio/language_steering/common_voice_fr_40203026.wav"
]

process_audio_files(audio_files, steering_module)
