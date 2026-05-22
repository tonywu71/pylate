"""Optional dispatch into ``late-interaction-kernels`` (LIK).

When LIK is installed and the input tensors live on a supported device, the
fused Triton (CUDA Ampere+) or ``torch.compile`` (Apple Silicon MPS) MaxSim
kernels are used. Otherwise the dispatcher returns ``None`` and the caller
falls back to pylate's reference einsum implementation.

The kill switches ``PYLATE_DISABLE_LIK=1`` and the legacy ``LIK_DISABLE=1``
force the fallback path; both are honored so existing user habits keep
working.
"""

from __future__ import annotations

import os

import torch

try:
    import late_interaction_kernels as _lik  # noqa: F401

    _LIK_AVAILABLE: bool = True
except ImportError:
    _LIK_AVAILABLE = False


def _is_disabled() -> bool:
    return (
        os.environ.get("PYLATE_DISABLE_LIK", "0") == "1"
        or os.environ.get("LIK_DISABLE", "0") == "1"
    )


def _device_path(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
) -> str | None:
    """Pick the LIK dispatch path or ``None`` to defer to pylate's reference.

    Returns ``"cuda"`` for the fused Triton kernel, ``"mps"`` for the
    ``torch.compile`` kernel, or ``None`` when LIK is unavailable, disabled,
    or the device/shape combination is unsupported.
    """
    if not _LIK_AVAILABLE or _is_disabled():
        return None
    if queries_embeddings.device != documents_embeddings.device:
        return None
    # LIK kernels assume embedding dim is large enough to vectorize over.
    if queries_embeddings.shape[-1] < 8:
        return None
    if queries_embeddings.is_cuda and documents_embeddings.is_cuda:
        capability: tuple[int, int] | None = None
        try:
            capability = torch.cuda.get_device_capability(queries_embeddings.device)
        except Exception:
            return None
        # bf16 + modern tensor cores require Ampere or newer.
        if capability[0] < 8:
            return None
        return "cuda"
    if (
        queries_embeddings.device.type == "mps"
        and documents_embeddings.device.type == "mps"
    ):
        return "mps"
    return None


def _mask_as_bool(mask: torch.Tensor | None) -> torch.Tensor | None:
    """pylate masks can be float (0/1), bool, or ``None``."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    return mask != 0


def maxsim_or_none(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None,
    documents_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the LIK MaxSim kernel or return ``None`` to fall back.

    pylate's ``colbert_scores`` does not L2-normalize internally (the encoder
    is expected to emit unit vectors), so we keep ``normalize=False`` on every
    backend.
    """
    path: str | None = _device_path(queries_embeddings, documents_embeddings)
    if path is None:
        return None

    q_mask: torch.Tensor | None = _mask_as_bool(queries_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(documents_mask)

    if path == "cuda":
        from late_interaction_kernels.autograd import maxsim

        return maxsim(
            queries_embeddings,
            documents_embeddings,
            q_mask=q_mask,
            d_mask=d_mask,
        )
    if path == "mps":
        from late_interaction_kernels.mps import maxsim_mps

        return maxsim_mps(
            queries_embeddings,
            documents_embeddings,
            q_mask=q_mask,
            d_mask=d_mask,
            normalize=False,
        )
    return None


def maxsim_kd_or_none(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None,
    documents_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """LIK dispatch for the knowledge-distillation scoring shape.

    KD layout: ``D`` is ``[Nq, Nd, Ld, d]`` — each query has its own candidate
    list — so we iterate one query at a time through the standard
    :func:`maxsim_or_none` kernel.
    """
    path: str | None = _device_path(queries_embeddings, documents_embeddings)
    if path is None:
        return None
    if documents_embeddings.dim() != 4:
        return None

    q_mask: torch.Tensor | None = _mask_as_bool(queries_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(documents_mask)

    num_queries: int = queries_embeddings.shape[0]
    num_docs: int = documents_embeddings.shape[1]
    out: torch.Tensor = torch.empty(
        num_queries,
        num_docs,
        device=queries_embeddings.device,
        dtype=torch.float32,
    )
    for query_index in range(num_queries):
        per_query_q_mask: torch.Tensor | None = (
            q_mask[query_index : query_index + 1] if q_mask is not None else None
        )
        per_query_d_mask: torch.Tensor | None = (
            d_mask[query_index] if d_mask is not None else None
        )
        per_query_score: torch.Tensor | None = maxsim_or_none(
            queries_embeddings[query_index].unsqueeze(0),
            documents_embeddings[query_index],
            per_query_q_mask,
            per_query_d_mask,
        )
        # ``maxsim_or_none`` returns None only if the dispatcher rejects the
        # input; the outer ``_device_path`` already accepted the batch, so a
        # per-query rejection here would indicate a shape mismatch — bail to
        # the fallback path rather than silently producing partial results.
        if per_query_score is None:
            return None
        out[query_index] = per_query_score.squeeze(0)
    return out
