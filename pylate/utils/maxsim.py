"""Dispatch helper for late-interaction (MaxSim) scoring.

``maxsim_inbatch`` and ``maxsim_kd`` are the entry points used by
:func:`pylate.scores.colbert.colbert_scores` and
:func:`pylate.scores.colbert.colbert_kd_scores`. They route through the fused
Triton (CUDA Ampere+) or ``torch.compile`` (Apple Silicon MPS) kernels from
``late-interaction-kernels`` when the dependency is installed and the runtime
is supported, and fall through to a pure-torch ``einsum + amax + sum``
reference otherwise.

The kill switches ``PYLATE_DISABLE_LIK=1`` and the legacy ``LIK_DISABLE=1``
force the reference path; both are honored so existing user habits keep
working.
"""

import os

import torch

try:
    import late_interaction_kernels as _lik  # noqa: F401

    _LIK_AVAILABLE: bool = True
except ImportError:
    _LIK_AVAILABLE = False


def _is_disabled() -> bool:
    """True when the user has set either kill-switch env var."""
    return (
        os.environ.get("PYLATE_DISABLE_LIK", "0") == "1"
        or os.environ.get("LIK_DISABLE", "0") == "1"
    )


def _mask_as_bool(mask: torch.Tensor | None) -> torch.Tensor | None:
    """pylate masks can be float (0/1), bool, or ``None``."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    return mask != 0


def _dispatch_path(query: torch.Tensor, doc: torch.Tensor) -> str | None:
    """Pick the dispatch backend or return ``None`` to fall back to torch.

    Returns ``"cuda"`` for CUDA Ampere+ devices, ``"mps"`` for Apple Silicon,
    or ``None`` for every other case. The bail-out conditions:

    * LIK not installed — there is no kernel to call.
    * ``PYLATE_DISABLE_LIK=1`` / ``LIK_DISABLE=1`` — manual override for A/B
      testing or numeric debugging without uninstalling LIK.
    * Mixed devices — the kernel runs on a single device; cross-device tensors
      would error or trigger an implicit copy.
    * ``d < 8`` — Triton tile / MMA shapes have a hard lower bound on the
      embedding dim; below it the kernel won't outperform einsum.
    * CUDA capability ``< 8`` (pre-Ampere) — lacks the bf16 tensor-core path
      the kernel autotunes for.
    * CPU or any other backend — no LIK kernel exists for it.
    """
    if not _LIK_AVAILABLE or _is_disabled():
        return None
    if query.device != doc.device:
        return None
    if query.shape[-1] < 8:
        return None
    if query.is_cuda and doc.is_cuda:
        # Need Ampere or newer for bf16 + modern tensor cores.
        if torch.cuda.get_device_capability(query.device)[0] < 8:
            return None
        return "cuda"
    if query.device.type == "mps" and doc.device.type == "mps":
        return "mps"
    return None


def _torch_maxsim(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None,
    doc_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reference MaxSim: ``einsum("ash,bth->abst").max(-1).sum(-1)``."""
    scores: torch.Tensor = torch.einsum("ash,bth->abst", query, doc)
    if query_mask is not None:
        scores = scores * query_mask.unsqueeze(1).unsqueeze(3)
    if doc_mask is not None:
        scores = scores * doc_mask.unsqueeze(0).unsqueeze(2)
    return scores.max(axis=-1).values.sum(axis=-1)


def _torch_maxsim_kd(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None,
    doc_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reference KD MaxSim: per-query candidate lists, ``doc`` is ``[Nq, Nd, Ld, d]``."""
    scores: torch.Tensor = torch.einsum("ash,abth->abst", query, doc)
    if query_mask is not None:
        scores = scores * query_mask.unsqueeze(1).unsqueeze(3)
    if doc_mask is not None:
        scores = scores * doc_mask.unsqueeze(2)
    return scores.max(axis=-1).values.sum(axis=-1)


def maxsim_inbatch(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch MaxSim scores for late-interaction scoring.

    Args:
        query: ``[Nq, Lq, d]`` query token embeddings.
        doc: ``[Nd, Ld, d]`` document token embeddings.
        query_mask: optional ``[Nq, Lq]`` boolean/float mask for query tokens.
        doc_mask: optional ``[Nd, Ld]`` boolean/float mask for doc tokens.

    Returns:
        ``[Nq, Nd]`` similarity matrix — the sum over query tokens of each
        token's max similarity against ``doc``'s token dimension.
    """
    path: str | None = _dispatch_path(query, doc)
    if path is None:
        return _torch_maxsim(query, doc, query_mask, doc_mask)

    q_mask: torch.Tensor | None = _mask_as_bool(query_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(doc_mask)

    if path == "cuda":
        from late_interaction_kernels.autograd import maxsim as _lik_maxsim

        return _lik_maxsim(query, doc, q_mask=q_mask, d_mask=d_mask)
    if path == "mps":
        from late_interaction_kernels.mps import maxsim_mps as _lik_maxsim_mps

        return _lik_maxsim_mps(query, doc, q_mask=q_mask, d_mask=d_mask, normalize=False)
    return _torch_maxsim(query, doc, query_mask, doc_mask)


def maxsim_kd(
    query: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Knowledge-distillation MaxSim: each query has its own candidate list.

    Args:
        query: ``[Nq, Lq, d]`` query token embeddings.
        doc: ``[Nq, Nd, Ld, d]`` per-query document token embeddings.
        query_mask: optional ``[Nq, Lq]`` mask.
        doc_mask: optional ``[Nq, Nd, Ld]`` mask.

    Returns:
        ``[Nq, Nd]`` similarity matrix.
    """
    path: str | None = _dispatch_path(query, doc)
    if path is None or doc.dim() != 4:
        return _torch_maxsim_kd(query, doc, query_mask, doc_mask)

    q_mask: torch.Tensor | None = _mask_as_bool(query_mask)
    d_mask: torch.Tensor | None = _mask_as_bool(doc_mask)

    num_queries: int = query.shape[0]
    num_docs: int = doc.shape[1]
    out: torch.Tensor = torch.empty(
        num_queries, num_docs, device=query.device, dtype=torch.float32
    )
    for query_index in range(num_queries):
        out[query_index] = maxsim_inbatch(
            query[query_index].unsqueeze(0),
            doc[query_index],
            q_mask[query_index : query_index + 1] if q_mask is not None else None,
            d_mask[query_index] if d_mask is not None else None,
        ).squeeze(0)
    return out
