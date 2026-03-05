import torch
import numpy as np
import soundfile as sf
from typing import Optional

# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
def bandwidth_to_n_q(bw_kbps):
    return {1.5: 2, 3: 4, 6: 8, 12: 16, 24: 32}[bw_kbps]

def n_q_to_bandwidth(n_q):
    return {2: 1.5, 4: 3, 8: 6, 16: 12, 32: 24}[n_q]
    

def load_wav_mono(path, target_sr=24000):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        raise ValueError(f"Expected {target_sr} Hz but got {sr}. Resample externally for now.")
    return torch.from_numpy(audio).unsqueeze(0)  # [1,T]

#=========================================
# for audio to tokens in different formats required ..

def tokens_TN_to_BQT(tokens_TN: torch.Tensor) -> torch.Tensor:
    # [T,N] -> [1,N,T]
    if tokens_TN.ndim != 2:
        raise ValueError(f"Expected [T,N], got {tuple(tokens_TN.shape)}")
    return tokens_TN.transpose(0, 1).unsqueeze(0).contiguous()

def tokens_BQT_to_TN(tokens_BQT: torch.Tensor) -> torch.Tensor:
    # [B,N,T] -> [T,N] using batch 0
    if tokens_BQT.ndim != 3:
        raise ValueError(f"Expected [B,N,T], got {tuple(tokens_BQT.shape)}")
    return tokens_BQT[0].transpose(0, 1).contiguous()

def tokens_BQT_to_QBT(tokens_BQT: torch.Tensor) -> torch.Tensor:
    # [B,N,T] -> [N,B,T]
    if tokens_BQT.ndim != 3:
        raise ValueError(f"Expected [B,N,T], got {tuple(tokens_BQT.shape)}")
    return tokens_BQT.permute(1, 0, 2).contiguous()

# -----
# and now the audio to token encoder

@torch.no_grad()
def encode_audio_to_tokens(audio, model, device, bandwidth, fmt: str = "TN", return_cpu: bool = False):
    """
    Encode audio -> tokens.

    fmt:
      - "TN":  [T_frames, n_q]      (recommended canonical)
      - "BQT": [B, n_q, T_frames]   (HF-friendly)
      - "QBT": [n_q, B, T_frames]   (for quantizer.decode)
    """
    import numpy as np
    import torch

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    audio = audio.float()

    # Force [B,C,T]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.ndim == 2:
        audio = audio.unsqueeze(1)
    elif audio.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected audio shape {tuple(audio.shape)}; need [T], [B,T], or [B,C,T].")

    audio = audio.to(device)
    model.eval()

    enc = model.encode(audio, bandwidth=bandwidth)
    codes = enc.audio_codes  # often [B,1,n_q,T]

    n_q=codes.shape[2]
    print(f'encode_audio_to_tokens with n_q={n_q}')

    # Normalize to BQT = [B,n_q,T]
    if codes.ndim == 4:
        tokens_BQT = codes[:, 0].contiguous()
    elif codes.ndim == 3:
        tokens_BQT = codes.contiguous()
    else:
        raise ValueError(f"Unexpected audio_codes shape {tuple(codes.shape)}")

    if fmt.upper() == "BQT":
        out = tokens_BQT
    elif fmt.upper() == "QBT":
        out = tokens_BQT_to_QBT(tokens_BQT)
    elif fmt.upper() == "TN":
        out = tokens_BQT_to_TN(tokens_BQT)
    else:
        raise ValueError(f"fmt must be 'TN', 'BQT', or 'QBT' (got {fmt})")

    if return_cpu and isinstance(out, torch.Tensor):
        out = out.cpu()

    return out, enc


#====================================================
# tokens_TN_to_audio_1T
#----------------------------------------------------

@torch.no_grad()
def tokens_TN_to_audio_1T(model, tokens_TN: torch.Tensor, device, audio_scales=None, last_frame_pad_length: int = 0):
    """
    tokens_TN: [T_frames, N] (TN format)
    Returns: audio_1T: [1, num_samples] on CPU (ready for IPython Audio)
    """
    model.eval()
    tokens_TN = tokens_TN.to(device, dtype=torch.long)

    # HF expects [B,1,N,T]
    codes_B1NT = tokens_TN.transpose(0, 1).unsqueeze(0).unsqueeze(0).contiguous()

    if audio_scales is None:
        audio_scales = [None]  # matches your .ecdc files

    audio_BCT = model.decode(
        audio_codes=codes_B1NT,
        audio_scales=audio_scales,
        last_frame_pad_length=last_frame_pad_length,
        padding_mask=None,
    )[0]  # [B,C,T]

    return audio_BCT[:, 0, :].detach().cpu().contiguous()


#====================================================
# read in a token coded file

def load_ecdc(path: str):
    """
    Loads your .ecdc (torch-saved dict with keys audio_codes/audio_scales/audio_length)
    and returns:
      tokens: LongTensor [T, N]
      scales: FloatTensor or None
      raw: the original dict
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict, got {type(obj)}")

    # Prefer your actual keys
    tokens = obj.get("audio_codes", None)
    scales = obj.get("audio_scales", None)

    if tokens is None:
        # fallback to other naming conventions
        tokens = obj.get("tokens", None) or obj.get("codes", None)
    if tokens is None:
        raise ValueError(f"Loaded {path} but couldn't find audio_codes/tokens/codes.")

    # tokens is (1, 1, N, T) in your file: (1,1,8,375)
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)

    if tokens.ndim == 4:
        # (B, ?, N, T) -> take [0,0] -> (N,T) -> transpose -> (T,N)
        tokens = tokens[0, 0]
        tokens = tokens.transpose(0, 1)  # (T,N)
    elif tokens.ndim == 3:
        # common alt: (B,N,T) -> [0] -> (N,T) -> transpose
        tokens = tokens[0].transpose(0, 1)
    elif tokens.ndim == 2:
        # (T,N) or (N,T) heuristic: if first dim small, it's (N,T)
        if tokens.shape[0] <= 64 and tokens.shape[1] > tokens.shape[0]:
            tokens = tokens.transpose(0, 1)
    else:
        raise ValueError(f"Unexpected tokens shape {tuple(tokens.shape)}")

    tokens = tokens.long().contiguous()

    # scales in your file is [None]
    if isinstance(scales, list):
        scales = scales[0] if len(scales) > 0 else None
    if scales is not None and not isinstance(scales, torch.Tensor):
        scales = torch.tensor(scales).float()

    return tokens, scales, obj

#=============================================================
#=============================================================
# This is for building the token->latent lookup table

# ---- Helper: make sure audio is [B,C,T] ----
def ensure_BCT(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio[None, None, :]
    if audio.ndim == 2:
        return audio[:, None, :]
    if audio.ndim == 3:
        return audio
    raise ValueError(f"Unexpected audio shape {tuple(audio.shape)}; expected [T], [B,T], or [B,C,T].")

# ---- Build lookup table by decoding each token index (RNeNcodec-style) ----
#  You only need to do this once!

@torch.no_grad()
def build_LOOKUP_via_layer_decode(model, n_q: int, K: int, device=None) -> torch.Tensor:
    """
    Builds LOOKUP of shape (n_q, K, 128) by calling layers[q].decode(idx_all).

    This matches your RNeNcodec method:
        idx_all: (K,1)
        layers[q].decode(idx_all) -> (K,128,1)
    """
    if device is None:
        device = next(model.parameters()).device

    q = getattr(model, "quantizer", None)
    layers = getattr(q, "layers", None) or getattr(getattr(q, "vq", None), "layers", None)
    if layers is None:
        raise RuntimeError("encodec_model.quantizer.layers not found")

    idx_all = torch.arange(K, device=device, dtype=torch.long).unsqueeze(1)  # (K,1)

    E_list = []
    for qidx in range(n_q):
        z_kD1 = layers[qidx].decode(idx_all)  # expected (K,128,1)
        if z_kD1.ndim != 3 or z_kD1.shape[-1] != 1:
            raise RuntimeError(f"layers[{qidx}].decode(idx_all) returned {tuple(z_kD1.shape)}, expected (K,128,1)")
        E_list.append(z_kD1.squeeze(-1).contiguous())  # (K,128)

    return torch.stack(E_list, dim=0)  # (n_q,K,128)

# ==========================================================================
# ---- Token stack (T,n_q) -> latent (T,128) via lookup+sum ----
# Can be used as a source "pool" of talents for style transfer purposes

@torch.no_grad()
def tokens_to_summary_latents(tokens_TN: torch.Tensor, LOOKUP_QKD: torch.Tensor) -> torch.Tensor:
    """
    tokens_TN: (T, n_q) long
    LOOKUP_QKD: (n_q, K, 128)
    returns:   (T, 128)
    """
    tokens_TN = tokens_TN.to(LOOKUP_QKD.device, dtype=torch.long)
    T, n_q = tokens_TN.shape
    print(f'Taking tokens_to_summary_latents with n_q={n_q}')
    assert LOOKUP_QKD.shape[0] >= n_q

    z_TD = torch.zeros(T, 128, device=LOOKUP_QKD.device, dtype=LOOKUP_QKD.dtype)
    for q in range(n_q):
        z_TD.add_(LOOKUP_QKD[q][tokens_TN[:, q]])
    return z_TD

# ---- Same as above, but only for one level of a token stack
@torch.no_grad()
def token_level_to_latents(tokens_TN: torch.Tensor, level_q: int, LOOKUP_QKD: torch.Tensor) -> torch.Tensor:
    tokens_TN = tokens_TN.to(LOOKUP_QKD.device, dtype=torch.long)
    return LOOKUP_QKD[level_q][tokens_TN[:, level_q]]  # (T,128)
    
# --------------------------------------------------------------------------
@torch.no_grad()
def audio_to_latents(audio, model, device: str, bandwidth):
    """
    audio -> quantized latents (summary latents)

    Returns:
      z_T128: FloatTensor [T_frames, 128]   (for batch=1)
      codes_TQ: LongTensor  [T_frames, n_q] (for batch=1)
      enc: encode output
    """
    #codes_BQT, enc = audio_to_codes(audio, model, device=device, bandwidth=bandwidth)

    codes_BQT, enc =  encode_audio_to_tokens(audio, model, device=device, bandwidth=bandwidth, fmt="BQT")
    

    # HF quantizer.decode expects (n_q, B, T)
    codes_QBT = codes_BQT.permute(1, 0, 2).contiguous()

    z_BDT = model.quantizer.decode(codes_QBT)     # [B,128,T]
    z_BTD = z_BDT.permute(0, 2, 1).contiguous()   # [B,T,128]

    if z_BTD.shape[0] != 1:
        # If you ever pass batch>1, return the full batch tensor instead
        return z_BTD, codes_BQT.permute(0, 2, 1).contiguous(), enc

    z_T128 = z_BTD[0]                              # [T,128]
    codes_TQ = codes_BQT[0].transpose(0, 1).contiguous()  # [T,n_q]
    return z_T128, codes_TQ, enc

#====================================================
# Latents to audio
#----------------------------------------------------
@torch.no_grad()
def latents128_to_audio(model, z_T128, device):
    """
    z_T128: [T_frames, 128] quantized latents (the thing you verified).
    Returns: audio_1T: [1, num_samples]
    """
    model.eval()
    z_T128 = z_T128.to(device)

    # decoder expects [B, D, T_frames]
    z_BDT = z_T128.unsqueeze(0).transpose(1, 2).contiguous()  # [1,128,T]

    # Most HF builds expose decoder as model.decoder(...)
    audio_BCT = model.decoder(z_BDT)  # [1,C,T_samples]
    return audio_BCT[:, 0, :].contiguous()




# =========================================================================
# For STREAMING tests
# =========================================================================
import math
import torch

@torch.no_grad()
def latents128_to_audio_streaming(
    model,
    z_T128: torch.Tensor,
    device,
    chunk_size: int,    # the number of frames to be decoded on reach hop (like fft "window size")
    hop_size: int,      # the number of frames to advance at each step (like fft - smaller than window size)
    prefix_T128: Optional[torch.Tensor] = None,
):
    """
    Streaming/sliding-window latent decode with optional *prefix padding* for alignment.

    Goal:
      - Output length matches the original input length (in frames -> samples).
      - The nth output frame corresponds to the nth input frame, even when chunk_size > hop_size,
        by conceptually providing (chunk_size - hop_size) frames of "past context" before frame 0.

    Args:
      model: HuggingFace EncodecModel (or compatible) with model.decoder(z_BDT)->audio_BCT
      z_T128: Tensor [T_frames, 128] quantized latents
      device: torch device / string
      chunk_size: window length in *latent frames*
      hop_size: hop length in *latent frames*
      prefix_T128:
        - If provided: Tensor [pad, 128] where pad = max(0, chunk_size - hop_size),
          prepended to z_T128 before streaming decode.
        - If None: the function repeats the first latent frame pad times.

    Returns:
      audio_1T: Tensor [1, T_frames * samples_per_frame] on the model/device dtype/device.

    Notes:
      - Uses the same decoder input layout as your existing function: [B, D, T_frames] with D=128.
      - Assumes model.config.sampling_rate and model.config.frame_rate exist and yield an integer
        samples_per_frame (= sr / frame_rate).
    """
    if z_T128.ndim != 2 or z_T128.shape[1] != 128:
        raise ValueError(f"Expected z_T128 with shape [T,128], got {tuple(z_T128.shape)}")
    if chunk_size <= 0 or hop_size <= 0:
        raise ValueError("chunk_size and hop_size must be positive integers")

    model.eval()
    z_T128 = z_T128.to(device)

    # Infer samples per latent frame
    cfg = getattr(model, "config", None)
    sr = getattr(cfg, "sampling_rate", None)
    fr = getattr(cfg, "frame_rate", None)
    if sr is None or fr is None:
        raise RuntimeError(
            "Could not infer samples_per_frame. Expected model.config.sampling_rate and "
            "model.config.frame_rate to exist."
        )

    samples_per_frame_f = sr / fr
    samples_per_frame = int(round(samples_per_frame_f))
    if not math.isclose(samples_per_frame_f, samples_per_frame, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(
            f"Non-integer samples_per_frame={samples_per_frame_f} (sr={sr}, frame_rate={fr})"
        )

    T_orig = z_T128.shape[0]
    target_samples = T_orig * samples_per_frame

    # Alignment padding
    pad = max(0, chunk_size - hop_size)
    if pad > 0:
        if prefix_T128 is None:
            z0 = z_T128[:1]  # [1,128]
            prefix = z0.repeat(pad, 1)  # [pad,128]
        else:
            if prefix_T128.ndim != 2 or prefix_T128.shape[1] != 128:
                raise ValueError(
                    f"prefix_T128 must have shape [{pad},128]; got {tuple(prefix_T128.shape)}"
                )
            if prefix_T128.shape[0] != pad:
                raise ValueError(
                    f"prefix_T128 length must be pad={pad} (chunk_size-hop_size). "
                    f"Got {prefix_T128.shape[0]}."
                )
            prefix = prefix_T128.to(device)
        z_stream = torch.cat([prefix, z_T128], dim=0)  # [pad+T,128]
    else:
        z_stream = z_T128

    T_stream = z_stream.shape[0]
    out_chunks = []

    start = 0
    while start < T_stream:
        end = min(start + chunk_size, T_stream)
        z_win = z_stream[start:end]  # [L,128]
        L = z_win.shape[0]

        # [B, D, T_frames] with D=128
        z_BDT = z_win.unsqueeze(0).transpose(1, 2).contiguous()  # [1,128,L]
        audio_BCT = model.decoder(z_BDT)                          # [1,C,T_samples]
        audio_1T = audio_BCT[:, 0, :].contiguous()                # [1,T_samples]

        remaining_frames = T_stream - start
        emit_frames = min(hop_size, remaining_frames)
        emit_samples = emit_frames * samples_per_frame

        if emit_samples > audio_1T.shape[1]:
            raise RuntimeError(
                f"emit_samples ({emit_samples}) > decoded samples ({audio_1T.shape[1]}). "
                f"Check samples_per_frame inference (sr={sr}, fr={fr})."
            )

        out_chunks.append(audio_1T[:, -emit_samples:])
        start += hop_size

    audio = torch.cat(out_chunks, dim=1)  # [1, >= target_samples]

    # Trim to exactly the original length (aligned to original frame 0)
    if audio.shape[1] < target_samples:
        raise RuntimeError(
            f"Decoded audio shorter than expected: got {audio.shape[1]} samples, "
            f"expected at least {target_samples}."
        )
    return audio[:, :target_samples]