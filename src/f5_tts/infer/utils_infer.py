# utils_infer.py

# A unified script for the inference process.
# Make adjustments inside functions; consider both gradio and CLI scripts if you need to change the output format.

import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

_ref_audio_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ------------------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# ------------------------------------------------


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """
    Splits the input text into chunks by breaking at spaces, creating visually balanced chunks.

    Args:
        text (str): The text to be split.
        max_chars (int): Approximate maximum number of bytes per chunk in UTF-8 encoding.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Replace spaces with <unk> temporarily, then split on <unk> or whitespace
    text = text.replace(" ", "<unk>")
    segments = re.split(r"(<unk>|\s+)", text)

    for segment in segments:
        if not segment or segment in ("<unk>", " "):
            continue
        # Check byte length in UTF-8
        if len((current_chunk + segment).encode("utf-8")) <= max_chars:
            current_chunk += segment + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = segment + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Replace <unk> back with real spaces
    return [chunk.replace("<unk>", " ") for chunk in chunks]


def load_vocoder(
    vocoder_name: str = "vocos",
    is_local: bool = False,
    local_path: str = "",
    device: str = device,
    hf_cache_dir: str = None,
) -> Vocos:
    """
    Load the Vocos (or BigVGAN) vocoder. Defaults to 'vocos' from HF hub.
    """
    if vocoder_name == "vocos":
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")

        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)

        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)

    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init BigVGAN submodule.")
            raise

        if is_local:
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)

    return vocoder


# ASR pipeline (used if no custom ref_text is provided)
asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


def transcribe(ref_audio: str, language: str = None) -> str:
    """
    Run Whisper ASR on `ref_audio` if no custom ref_text was given.
    """
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return (
        asr_pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe", "language": language}
            if language
            else {"task": "transcribe"},
            return_timestamps=False,
        )["text"]
        .strip()
    )


def load_checkpoint(model, ckpt_path: str, device: str, dtype=None, use_ema: bool = True):
    """
    Load a checkpoint into a CFM model (supports safetensors or .pt).
    """
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # remove any deprecated keys
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()
    return model.to(device)


def load_model(
    model_cls,
    model_cfg: dict,
    ckpt_path: str,
    mel_spec_type: str = mel_spec_type,
    vocab_file: str = "",
    ode_method: str = ode_method,
    use_ema: bool = True,
    device: str = device,
):
    """
    Instantiate and load a CFM model for inference.
    """
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab :", vocab_file)
    print("token :", tokenizer)
    print("model :", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)
    return model


def remove_silence_edges(audio: AudioSegment, silence_threshold: int = -42) -> AudioSegment:
    """
    Remove leading/trailing silence from a pydub AudioSegment.
    """
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence at the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]
    return trimmed_audio


def preprocess_ref_audio_text(
    ref_audio_orig: str,
    ref_text: str,
    clip_short: bool = True,
    show_info=print,
    device: str = device,
) -> tuple[str, str]:
    """
    Convert reference audio to a temporary .wav, apply silence removal/clipping,
    optionally run ASR if no ref_text was given, and ensure it ends with proper punctuation.
    Returns (ref_audio_path, ref_text).
    """
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            # 3. if still too long, forcibly trim
            if len(non_silent_wave) > 15000:
                non_silent_wave = non_silent_wave[:15000]
                show_info("Audio is over 15s, clipping short. (3)")

            aseg = non_silent_wave

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        asegur.export(f.name, format="wav")
        ref_audio = f.name

    # Compute MD5 hash
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided; running ASR...")
            ref_text = transcribe(ref_audio)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure trailing punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)
    return ref_audio, ref_text


def infer_process(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    model_obj,
    vocoder,
    mel_spec_type: str = mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms: float = target_rms,
    cross_fade_duration: float = cross_fade_duration,
    nfe_step: int = nfe_step,
    cfg_strength: float = cfg_strength,
    sway_sampling_coef: float = sway_sampling_coef,
    speed: float = speed,
    fix_duration=None,
    device: str = device,
    set_max_chars: int = 250,
):
    """
    Split the input text into batches, respecting newline first.
      - If gen_text contains newlines, split by line.
      - For each line, if it’s longer than set_max_chars, further chunk it via chunk_text().
    Then pass each chunk into infer_batch_process.
    """
    # 1. Respect newline splitting first
    lines = gen_text.splitlines()
    gen_text_batches = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # If short enough, keep as-is
        if len(line.encode("utf-8")) <= set_max_chars:
            gen_text_batches.append(line)
        else:
            # Otherwise break into smaller sub-chunks
            subchunks = chunk_text(line, max_chars=set_max_chars)
            gen_text_batches.extend(subchunks)

    # Print each chunk
    for i, batch_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", batch_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")

    # 2. Load the reference waveform + sample rate correctly via torchaudio
    waveform, sr = torchaudio.load(ref_audio)

    return next(
        infer_batch_process(
            (waveform, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


def infer_batch_process(
    ref_audio: tuple[torch.Tensor, int],
    ref_text: str,
    gen_text_batches: list[str],
    model_obj,
    vocoder,
    mel_spec_type: str = "vocos",
    progress=tqdm,
    target_rms: float = 0.1,
    cross_fade_duration: float = 0.15,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1,
    speed: float = 1,
    fix_duration=None,
    device=None,
    streaming: bool = False,
    chunk_size: int = 2048,
):
    """
    Actually run the diffusion-based model + vocoder on each text chunk in parallel (threads), then
    either chain them (with optional cross-fading) or return streaming chunks.
    """
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text: str):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare [ref_text + gen_text] → pinyin
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # Diffusion-sample
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)

            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            else:  # bigvgan
                generated_wave = vocoder(generated_mel_spec)

            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j: j + chunk_size], target_sample_rate
            else:
                yield generated_wave, generated_mel_spec[0].cpu().numpy()

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                final_wave = np.concatenate(generated_waves)
            else:
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )
                    final_wave = new_wave

            combined_spectrogram = np.concatenate(spectrograms, axis=1)
            yield final_wave, target_sample_rate, combined_spectrogram
        else:
            yield None, target_sample_rate, None


def remove_silence_for_generated_wav(filename: str):
    """
    Given a filename of a generated .wav, strip out any leading/trailing silence.
    """
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    non_silent_wave.export(filename, format="wav")


def save_spectrogram(spectrogram, path: str):
    """
    Helper to save a spectrogram image to disk.
    """
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
