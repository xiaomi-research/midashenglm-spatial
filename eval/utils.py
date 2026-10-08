"""Inference utilities for MiDashengLM-Spatial.

Provides helpers to load the model/processor, read (possibly binaural) audio,
and run single-sample generation.

Example:
    model, processor, device = load_model("mispeech/midashenglm-spatial")
    text = "Describe the spatial layout of the sounds."
    print(generate("binaural.wav", text, model, processor, device))
"""

from typing import List, Sequence, Tuple, Union

# import librosa
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoProcessor

DEFAULT_MAX_SEC = 600  # clip individual audio files to at most 10 minutes
SAMPLE_RATE = 16000  # model's native sampling rate
NUM_CHANNELS = 2  # binaural input
GAP_SEC = 1.0  # seconds of silence to add between audio files


def load_model(
    model_path: str,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, AutoProcessor, torch.device]:
    """Load the model and its processor onto the best available device.

    Args:
        model_path: HF repo id or local directory of the model package.
        dtype: Weight dtype; use ``torch.bfloat16`` on GPU to save memory.

    Returns:
        A ``(model, processor, device)`` tuple, with the model in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, device


def load_audio(audio_path: Union[str, Sequence[str]]) -> np.ndarray:
    """Load one or more audio files as a ``(NUM_CHANNELS, n_samples)`` array.

    A single path or a sequence of paths may be given. Each file is loaded at
    ``SAMPLE_RATE``, capped at ``DEFAULT_MAX_SEC`` seconds, and coerced to
    ``NUM_CHANNELS`` channels (mono is up-mixed by repetition; extra channels are
    truncated to the first channel and then up-mixed). Multiple files are
    concatenated in order with ``GAP_SEC`` seconds of silence between them.
    """

    def expand_to_binaural(waveform: np.ndarray) -> np.ndarray:
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)  # (n,) -> (1, n)
        if waveform.shape[0] != NUM_CHANNELS:
            waveform = waveform[:1]  # keep the first channel
        if waveform.shape[0] == 1:
            waveform = np.repeat(waveform, NUM_CHANNELS, axis=0)  # up-mix mono -> binaural
        return waveform

    if isinstance(audio_path, str):
        audio_path = [audio_path]
    max_len = int(DEFAULT_MAX_SEC * SAMPLE_RATE)
    gap = np.zeros((NUM_CHANNELS, int(GAP_SEC * SAMPLE_RATE)), dtype=np.float32)
    segments: List[np.ndarray] = []
    for i, path in enumerate(audio_path):
        # waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.numpy()
        if waveform.shape[-1] > max_len:
            waveform = waveform[..., :max_len]  # cap overly long clips
        waveform = expand_to_binaural(waveform)
        if i > 0:
            segments.append(gap)  # insert silence between consecutive clips
        segments.append(waveform)
    return np.concatenate(segments, axis=1)


def _format_msg(
    audio: Union[str, Sequence[str], np.ndarray, torch.Tensor], text: str
) -> list:
    """Build a single-turn chat message from an audio input and a text prompt.

    ``audio`` may be an in-memory array/tensor, or a path (or list of paths),
    which is loaded and concatenated here via :func:`load_audio`.
    """
    if not isinstance(audio, (np.ndarray, torch.Tensor)):
        audio = load_audio(audio)
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": text},
            ],
        }
    ]


@torch.inference_mode()
def generate(
    audio: Union[str, Sequence[str], np.ndarray, torch.Tensor],
    text: str,
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    max_new_tokens: int = 512,
) -> str:
    """Greedily generate a response for one (audio, text) sample.

    Args:
        audio: Audio array/tensor, or a path (or list of paths) to be loaded.
        text: Text prompt to accompany the audio.
        model: Model returned by :func:`load_model`.
        processor: Processor returned by :func:`load_model`.
        device: Target device for the inputs.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The decoded response. 
    """
    messages = _format_msg(audio, text)  # single conversation
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=True,
        return_dict=True,
    ).to(device)
    generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = processor.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    return response
