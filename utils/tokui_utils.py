import torch
import torch.nn.functional as F

@torch.no_grad()
def normalize_latents(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Row-normalize a [T,D] latent sequence for cosine similarity.
    """
    if x.ndim != 2:
        raise ValueError(f"normalize_latents expects [T,D], got {tuple(x.shape)}")
    return F.normalize(x.float(), dim=1, eps=eps)


@torch.no_grad()
def target2poolindex(
    target: torch.Tensor,
    pool_norm: torch.Tensor,
    window: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Windowed cosine matching from target frames to pool frames.

    Args
    ----
    target    : [T,D] float (unnormalized ok)  -- normalized internally
    pool_norm : [P,D] float                    -- MUST already be normalized row-wise
    window    : int >= 1. If >1, match non-overlapping windows of length `window`.
               Returns one pool index per target frame, enforcing sequential indices
               within each matched window.
    eps       : normalization epsilon

    Returns
    -------
    idx_T : [T] long. For each target frame t, idx_T[t] is the matched pool frame index.

    Notes
    -----
    - If window > 1, we find best pool start p for each target window and then set:
        idx[t + i] = p + i  for i in [0..window-1]
      so indices are sequential in the pool within each window.
    - If T is not divisible by window, we print a warning and handle the remainder
      with a shorter final window (still sequential in the pool).
    """
    if target.ndim != 2 or pool_norm.ndim != 2:
        raise ValueError(f"target and pool_norm must be [*,D]. Got {target.shape=} {pool_norm.shape=}")
    if target.shape[1] != pool_norm.shape[1]:
        raise ValueError(f"Dim mismatch: target D={target.shape[1]} vs pool D={pool_norm.shape[1]}")
    if window < 1:
        raise ValueError("window must be >= 1")

    device = target.device
    pool_norm = pool_norm.to(device)

    # Normalize target inside
    target_norm = normalize_latents(target, eps=eps).to(device)

    T, D = target_norm.shape
    P = pool_norm.shape[0]

    if window > P:
        raise ValueError(f"window={window} is larger than pool length P={P}")

    if window != 1 and (T % window) != 0:
        print(f"WARNING: window={window} does not evenly divide target length T={T}. "
              f"Handling final remainder of {T % window} frames with a shorter window.")

    # Precompute transpose for fast sims: [D,P]
    pool_T = pool_norm.transpose(0, 1).contiguous()

    idx_out = torch.empty(T, dtype=torch.long, device=device)

    def best_start_for_window(win_LD: torch.Tensor) -> int:
        """
        win_LD: [L,D] normalized
        returns best pool start p in [0, P-L]
        """
        L = win_LD.shape[0]
        P_L = P - L + 1  # number of valid start positions
        # sims: [L,P] = win @ pool^T (cos sims because both normalized)
        sims = win_LD @ pool_T  # [L,P]

        # Build indices to pick diagonal-aligned elements sims[i, p+i]
        # idx_mat: [P_L, L] with idx_mat[p, i] = p + i
        base = torch.arange(P_L, device=device).unsqueeze(1)           # [P_L,1]
        offs = torch.arange(L, device=device).unsqueeze(0)             # [1,L]
        idx_mat = base + offs                                          # [P_L,L]

        # Gather: for each start p, take sims[i, p+i] across i and sum
        rows = torch.arange(L, device=device).unsqueeze(0).expand(P_L, L)  # [P_L,L]
        gathered = sims[rows, idx_mat]                                     # [P_L,L]
        scores = gathered.sum(dim=1)                                       # [P_L]

        return int(scores.argmax().item())

    # Process full windows
    t = 0
    while t + window <= T:
        win = target_norm[t:t+window]          # [window,D]
        p_best = best_start_for_window(win)    # start in pool
        # sequential indices within window
        idx_out[t:t+window] = torch.arange(p_best, p_best + window, device=device, dtype=torch.long)
        t += window

    # Remainder window (if any)
    rem = T - t
    if rem > 0:
        win = target_norm[t:T]                 # [rem,D]
        p_best = best_start_for_window(win)
        idx_out[t:T] = torch.arange(p_best, p_best + rem, device=device, dtype=torch.long)

    return idx_out


#====================================================
# Latents to audio
#----------------------------------------------------
@torch.no_grad()
def latents_T128_to_audio(model, z_T128, device):
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

#=======================================================
# and now lets use our index into the pool to get back our new sequence
# ======================================================

@torch.no_grad()
def gather_tokens_by_index(tokens_TN: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    tokens_TN: [T_pool, N] on CPU (typical)
    idx:       [T_target] may be on GPU or CPU

    Returns: [T_target, N] on same device as tokens_TN
    """
    if tokens_TN.ndim != 2:
        raise ValueError(f"tokens_TN must be [T,N], got {tuple(tokens_TN.shape)}")

    idx = idx.view(-1).long()

    # Make idx compatible with tokens_TN device
    if tokens_TN.device.type == "cpu":
        idx = idx.cpu()
    else:
        idx = idx.to(tokens_TN.device)

    if idx.max().item() >= tokens_TN.shape[0] or idx.min().item() < 0:
        raise ValueError("idx contains out-of-range values for tokens_TN")

    return tokens_TN[idx]
#------------

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


########################################################################
#########################################################################
########################################################################
import math
import numpy as np
from typing import Union


from realtime_synth.generators.base import BaseGenerator
from realtime_synth.utils import exp_map01
from realtime_synth_ui import build_synth_ui

# Assumes these already exist in your codebase:
# - BaseGenerator
# - normalize_latents
# - target2poolindex


class PoolMatcherGenerator(BaseGenerator):
    param_labels = ["Pool start", "Pool end"]

    def __init__(
        self,
        model,
        target_latents_TD: torch.Tensor,
        pool_latents_TD: torch.Tensor, # RAW pool latents [P,128]
        *,
        init_norm_params=None,
        hop_size: int = 6,             # latent frames emitted per step
        chunk_size: int = 24,          # latent frames decoded per step
        window_size: int = 6,          # matching window length in latent frames
        device: Union[str, torch.device] = "cpu",
        wrap_target: bool = True,      # default: loop target forever
        lookup_qkd = None,             # stored for future use; not needed yet in this latent-only version
    ):
        super().__init__(init_norm_params or [0.0, 1.0])

        # --- basic checks ---
        if target_latents_TD.ndim != 2 or target_latents_TD.shape[1] != 128:
            raise ValueError(f"target_latents_TD must be [T,128], got {tuple(target_latents_TD.shape)}")
        if pool_latents_TD.ndim != 2 or pool_latents_TD.shape[1] != 128:
            raise ValueError(f"pool_latents_TD must be [P,128], got {tuple(pool_latents_TD.shape)}")

        if hop_size <= 0 or chunk_size <= 0 or window_size <= 0:
            raise ValueError("hop_size, chunk_size, and window_size must be positive")
        if chunk_size < hop_size:
            raise ValueError("chunk_size must be >= hop_size")
        if hop_size % window_size != 0:
            raise ValueError("hop_size must be an integer multiple of window_size")

        self.model = model
        self.lookup_qkd = lookup_qkd
        self.device = torch.device(device)
        self.model.eval()

        # store target + pool
        self.target_latents_TD = target_latents_TD.detach().float().to(self.device)
        self.pool_latents_raw_TD = pool_latents_TD.detach().float().to(self.device)
        self.pool_latents_norm_TD = normalize_latents(self.pool_latents_raw_TD)

        self.T_target = self.target_latents_TD.shape[0]
        self.T_pool = self.pool_latents_raw_TD.shape[0]

        self.hop_size = int(hop_size)
        self.chunk_size = int(chunk_size)
        self.window_size = int(window_size)
        self.pad = max(0, self.chunk_size - self.hop_size)
        self.wrap_target = bool(wrap_target)

        # infer model timing
        cfg = getattr(self.model, "config", None)
        sr = getattr(cfg, "sampling_rate", None)
        fr = getattr(cfg, "frame_rate", None)
        if sr is None or fr is None:
            raise RuntimeError(
                "Could not infer samples_per_frame. Expected model.config.sampling_rate and model.config.frame_rate."
            )

        spf = sr / fr
        self.samples_per_frame = int(round(spf))
        if not math.isclose(spf, self.samples_per_frame, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(f"Non-integer samples_per_frame={spf} (sr={sr}, frame_rate={fr})")

        self.sr_model = int(sr)
        self.frame_rate_model = float(fr)
        self.hop_samples = self.hop_size * self.samples_per_frame

        # runtime state
        self.target_cursor = 0
        self.audio_fifo = np.zeros(0, dtype=np.float32)
        self.chunk_latents_TD = None   # [chunk_size,128], initialized lazily

        self.set_params(self.norm_params)

    def set_params(self, norm_params):
        super().set_params(norm_params)

        # two sliders define normalized pool endpoints
        a = float(np.clip(self.norm_params[0], 0.0, 1.0))
        b = float(np.clip(self.norm_params[1], 0.0, 1.0))
        lo = min(a, b)
        hi = max(a, b)

        # map to integer indices
        p0 = int(round(lo * (self.T_pool - 1)))
        p1 = int(round(hi * (self.T_pool - 1)))

        # enforce enough room for at least one usable segment
        min_len = max(self.window_size, self.hop_size)
        if (p1 - p0 + 1) < min_len:
            p1 = min(self.T_pool - 1, p0 + min_len - 1)
            p0 = max(0, p1 - min_len + 1)

        self.pool_start = int(p0)
        self.pool_end = int(p1)

    def formatted_readouts(self):
        return [
            f"{self.param_labels[0]}: {self.pool_start:6d}",
            f"{self.param_labels[1]}: {self.pool_end:6d}",
        ]

    def reset(self):
        self.target_cursor = 0
        self.audio_fifo = np.zeros(0, dtype=np.float32)
        self.chunk_latents_TD = None

    def _current_pool_views(self):
        """
        Returns:
            pool_norm_sel: [Psel,128] normalized, for matching
            pool_raw_sel:  [Psel,128] raw, for playback
        """
        sl = slice(self.pool_start, self.pool_end + 1)
        return self.pool_latents_norm_TD[sl], self.pool_latents_raw_TD[sl]

    def _next_target_hop(self) -> torch.Tensor:
        """
        Returns next target hop: [hop_size,128]
        Wraps around if wrap_target=True.
        """
        if self.wrap_target:
            out = []
            need = self.hop_size
            while need > 0:
                remain = self.T_target - self.target_cursor
                take = min(need, remain)
                out.append(self.target_latents_TD[self.target_cursor:self.target_cursor + take])
                self.target_cursor = (self.target_cursor + take) % self.T_target
                need -= take
            return torch.cat(out, dim=0)

        # non-wrapping mode
        end = min(self.target_cursor + self.hop_size, self.T_target)
        hop = self.target_latents_TD[self.target_cursor:end]
        self.target_cursor = end

        if hop.shape[0] < self.hop_size:
            if hop.shape[0] > 0:
                pad = hop[-1:].repeat(self.hop_size - hop.shape[0], 1)
            else:
                pad = torch.zeros(self.hop_size, 128, device=self.device, dtype=self.target_latents_TD.dtype)
            hop = torch.cat([hop, pad], dim=0)

        return hop

    def transform_target_hop(self, hop_TD: torch.Tensor) -> torch.Tensor:
        """
        Hook for later creative processing (noise, interpolation, etc).
        For now: identity.
        """
        return hop_TD

    def _match_hop(self, target_hop_TD: torch.Tensor) -> torch.Tensor:
        """
        Match one target hop against the selected pool segment.
        Returns matched RAW playback latents [hop_size,128].
        """
        pool_norm_sel, pool_raw_sel = self._current_pool_views()

        idx_rel = target2poolindex(
            target_hop_TD,
            pool_norm_sel,
            window=self.window_size,
        )  # [hop_size], relative to selected pool slice

        return pool_raw_sel[idx_rel]  # [hop_size,128]

    @torch.no_grad()
    def _prime_chunk(self):
        """
        Initialize chunk_latents_TD with:
            [prefix repeated first matched frame] + [first matched hop]
        so chunk length is exactly chunk_size.
        """
        hop = self.transform_target_hop(self._next_target_hop())
        matched = self._match_hop(hop)  # [hop_size,128]

        if self.pad > 0:
            prefix = matched[:1].repeat(self.pad, 1)  # [pad,128]
            self.chunk_latents_TD = torch.cat([prefix, matched], dim=0)
        else:
            self.chunk_latents_TD = matched

        if self.chunk_latents_TD.shape != (self.chunk_size, 128):
            raise RuntimeError(
                f"Primed chunk shape {tuple(self.chunk_latents_TD.shape)} != ({self.chunk_size},128)"
            )

    @torch.no_grad()
    def _produce_one_hop_audio(self) -> np.ndarray:
        """
        Produce exactly one hop's worth of audio samples.
        Decode the full chunk, then emit only the last hop of decoded audio.
        """
        if self.chunk_latents_TD is None:
            self._prime_chunk()
        else:
            hop = self.transform_target_hop(self._next_target_hop())
            matched = self._match_hop(hop)  # [hop_size,128]

            # shift chunk by hop and append new matched hop
            self.chunk_latents_TD = torch.cat(
                [self.chunk_latents_TD[self.hop_size:], matched],
                dim=0
            )

        # decode whole chunk
        z_BDT = self.chunk_latents_TD.unsqueeze(0).transpose(1, 2).contiguous()  # [1,128,chunk_size]
        audio_BCT = self.model.decoder(z_BDT)                                     # [1,C,samples]
        audio_1T = audio_BCT[:, 0, :].contiguous()                                # [1,samples]

        if audio_1T.shape[1] < self.hop_samples:
            raise RuntimeError(
                f"Decoded chunk shorter than hop: got {audio_1T.shape[1]} samples, need {self.hop_samples}"
            )

        hop_audio = audio_1T[:, -self.hop_samples:].detach().cpu().numpy()[0]
        return hop_audio.astype(np.float32, copy=False)

    def generate(self, frames, sr):
        """
        RTPySynth callback.
        Returns exactly `frames` float32 mono samples.
        """
        if sr != self.sr_model:
            raise ValueError(f"Generator expects sr={self.sr_model}, got sr={sr}")

        if frames <= 0:
            return np.zeros(0, dtype=np.float32)

        out = np.empty(frames, dtype=np.float32)
        filled = 0

        while filled < frames:
            if self.audio_fifo.size == 0:
                self.audio_fifo = self._produce_one_hop_audio()

            take = min(frames - filled, self.audio_fifo.size)
            out[filled:filled + take] = self.audio_fifo[:take]
            self.audio_fifo = self.audio_fifo[take:]
            filled += take

        return out
    