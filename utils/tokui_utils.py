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



    