import pathlib
import warnings
import random
from collections import defaultdict
from typing import Callable, Optional
from torchaudio.transforms import MelSpectrogram

import fire
import torch
import torchaudio
import numpy as np
import whisper
from whisper_interpretability.global_utils import device, BaseActivationModule
from whisper_interpretability import whisper_repo

from jaxtyping import Float
from torch import Tensor
import os

OUT_DIR = "out/"
LANG_CODE = "fr"

"""
Save the mean activation for all layers in [activations_to_cache] to disk
Used for activation steering eg save the mean french activation and use it to 'steer' german text
"""
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


def get_mels_from_audio_path(
    audio_path: str, start_time_s: Optional[float] = None, end_time_s: Optional[float] = None
):
    audio = whisper.load_audio(audio_path)
    if start_time_s is not None and end_time_s is not None:
        audio = trim_audio(audio, start_time_s, end_time_s)
    audio = whisper_repo.pad_or_trim(audio.flatten())
    mels = torch.tensor(whisper_repo.log_mel_spectrogram(audio)).to(device)
    return mels

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root="french_dataset", url="french_clean", return_mels=False):
        super().__init__()
        if not os.path.exists(f"{root}/LibriSpeech/{url}"):
            download = True
        else:
            download = False
        try:
            self.dataset = torchaudio.datasets.LIBRISPEECH(download=download, url=url, root=root)
        except RuntimeError:
            print("Downloading dataset")
            self.dataset = torchaudio.datasets.LIBRISPEECH(download=download, url=url, root=root)
        self.root = root
        self.return_mels = return_mels

    def __getitem__(self, idx):
        idx = int(random.random() * len(self.dataset))  # randomly sample
        audio_path, sample_rate, transcript, *_ = self.dataset.get_metadata(idx)
        if self.return_mels:
            mels = get_mels_from_audio_path(audio_path=f"{self.root}/LibriSpeech/{audio_path}")
            return mels, "en", f"{self.root}/LibriSpeech/{audio_path}"
        else:
            return f"{self.root}/LibriSpeech/{audio_path}"

    def __len__(self):
        return len(self.dataset)

class FrenchDataset(torch.utils.data.Dataset):
    def __init__(self, root="speech/cv-corpus-18.0-delta-2024-06-14/fr", return_mels=False):
        super().__init__()
        try:
            # Load the CommonVoice dataset for the specified language
            self.dataset = torchaudio.datasets.COMMONVOICE(root=root, tsv="validated.tsv")
        except RuntimeError:
            print("Please download the dataset")

        self.root = root
        self.return_mels = return_mels
        self.mel_transform = MelSpectrogram()

    def __getitem__(self, idx):
        # Randomly sample an index
        idx = int(random.random() * len(self.dataset))

        # Get the metadata for the selected index
        audio_path, sample_rate, transcript, *_ = self.dataset[idx]
        print('passed audio path', audio_path)
        if self.return_mels:
            # Convert audio to mel spectrogram
            mels = get_mels_from_audio_path(audio_path=audio_path)
            return mels, audio_path
        else:
            # Return raw audio path
            return audio_path

    def __len__(self):
        return len(self.dataset)

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

def get_activations(
    activations_to_cache: list = [
        "encoder.blocks.0",
        "encoder.blocks.1",
        "encoder.blocks.2",
        "encoder.blocks.3",
    ],
    num_samples: int = 100,
    class_label: str = LANG_CODE,
    model_name: str = "tiny",
    sql_path: str = f"/home/ellenar/probes/just_{LANG_CODE}_val.sql",
    batch_size: int = 50,
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")
    actv_cache = WhisperActivationCache(
        model_name=model_name, activations_to_cache=activations_to_cache
    )
    # dataset = MultiClassDataset(
    #     num_entries=num_samples, class_labels=[class_label], sql_path=sql_path
    # )
    dataset = FrenchDataset(return_mels=True)
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    all_actvs = defaultdict(list)  # layer_name: activations
    for i, (data, *labels) in enumerate(dataloader):
        actv_cache.reset_state()
        actv_cache.forward(data.to(device))
        for layer in activations_to_cache:
            actvs = actv_cache.activations[f"{layer}"].to(dtype=torch.float32).to("cpu")
            all_actvs[layer].append(actvs)
        if i >= num_samples / batch_size:
            break
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for layer, actvs in all_actvs.items():
        actvs = torch.cat(actvs, dim=0).mean(
            dim=0
        )  # mean over all batches but retain sequence length
        torch.save(actvs, f"{OUT_DIR}/{model_name}_{layer}_{class_label}")
        print(f"Saved {actvs.shape} activations for layer {layer} to disk")
