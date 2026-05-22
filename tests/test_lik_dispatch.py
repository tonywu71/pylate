"""Tests for the optional late-interaction-kernels (LIK) dispatcher.

These tests exercise pylate/scores/_lik.py. The CPU paths always defer to
the einsum fallback, so the bulk of the suite verifies the dispatcher's
gating behavior (env-var kill switches, device checks, embedding-dim
floor). The CUDA parity test is skipped unless a compatible GPU is present.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
import torch

from pylate.scores import _lik, colbert_kd_scores, colbert_scores


@pytest.fixture
def clear_lik_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip the LIK kill-switch env vars for the duration of a test."""
    monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
    monkeypatch.delenv("LIK_DISABLE", raising=False)
    yield


class TestDispatcherGating:
    """`_device_path` and `maxsim_or_none` must return None whenever the LIK
    kernels can't (or shouldn't) run."""

    def test_cpu_inputs_return_none(self, clear_lik_env: None) -> None:
        queries = torch.randn(2, 4, 16)
        documents = torch.randn(3, 5, 16)
        assert _lik._device_path(queries, documents) is None
        assert (
            _lik.maxsim_or_none(queries, documents, None, None) is None
        )

    def test_low_embedding_dim_returns_none(self, clear_lik_env: None) -> None:
        # H < 8 → reject, even on a supported device.
        queries = torch.randn(2, 4, 4)
        documents = torch.randn(3, 5, 4)
        assert _lik._device_path(queries, documents) is None

    def test_mixed_devices_return_none(self, clear_lik_env: None) -> None:
        queries = torch.randn(2, 4, 16)
        documents = torch.randn(3, 5, 16).to(memory_format=torch.contiguous_format)
        # Fake mismatched device types by mocking one tensor's .device.
        # Simpler: just confirm CPU+CPU is None (already covered) and that
        # the mixed-device branch is reachable — we test it directly with a
        # meta-tensor stand-in.
        meta_documents = torch.empty_like(documents, device="meta")
        assert _lik._device_path(queries, meta_documents) is None

    @pytest.mark.parametrize("flag", ["PYLATE_DISABLE_LIK", "LIK_DISABLE"])
    def test_kill_switch_disables_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
    ) -> None:
        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        monkeypatch.setenv(flag, "1")
        queries = torch.randn(2, 4, 16)
        documents = torch.randn(3, 5, 16)
        assert _lik._device_path(queries, documents) is None


class TestMaskNormalization:
    """`_mask_as_bool` accepts pylate's float/bool/None mask flavors."""

    def test_none_passthrough(self) -> None:
        assert _lik._mask_as_bool(None) is None

    def test_bool_passthrough(self) -> None:
        mask = torch.tensor([[True, False, True]])
        out = _lik._mask_as_bool(mask)
        assert out is mask
        assert out.dtype == torch.bool

    def test_float_converts_to_bool(self) -> None:
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        out = _lik._mask_as_bool(mask)
        assert out.dtype == torch.bool
        assert out.tolist() == [[True, False, True]]


class TestEndToEndFallbackParity:
    """With LIK installed but CPU inputs, the dispatcher must return None and
    the einsum fallback must produce identical results to a kill-switched run.

    This guards against regressions where the dispatcher accidentally takes
    the CPU into a non-fallback path (e.g., by forgetting a device check).
    """

    def test_colbert_scores_cpu_matches_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        queries = torch.randn(2, 4, 16)
        documents = torch.randn(3, 5, 16)
        queries_mask = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
        )
        documents_mask = torch.ones(3, 5)

        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        with_dispatcher = colbert_scores(
            queries, documents, queries_mask, documents_mask
        )

        monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
        forced_fallback = colbert_scores(
            queries, documents, queries_mask, documents_mask
        )

        torch.testing.assert_close(with_dispatcher, forced_fallback)

    def test_colbert_kd_scores_cpu_matches_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(1)
        num_queries, num_docs = 3, 4
        queries = torch.randn(num_queries, 5, 16)
        documents = torch.randn(num_queries, num_docs, 7, 16)
        queries_mask = torch.ones(num_queries, 5)
        documents_mask = torch.ones(num_queries, num_docs, 7)

        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        with_dispatcher = colbert_kd_scores(
            queries, documents, queries_mask, documents_mask
        )

        monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
        forced_fallback = colbert_kd_scores(
            queries, documents, queries_mask, documents_mask
        )

        torch.testing.assert_close(with_dispatcher, forced_fallback)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestCudaParity:
    """When CUDA is available the LIK Triton kernel must agree with the einsum
    reference within float tolerance. Skipped on CPU-only runners."""

    def test_colbert_scores_cuda_matches_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        queries = torch.randn(4, 8, 32, device="cuda")
        documents = torch.randn(5, 12, 32, device="cuda")
        queries_mask = torch.ones(4, 8, device="cuda")
        documents_mask = torch.ones(5, 12, device="cuda")

        monkeypatch.delenv("PYLATE_DISABLE_LIK", raising=False)
        monkeypatch.delenv("LIK_DISABLE", raising=False)
        fused = colbert_scores(
            queries, documents, queries_mask, documents_mask
        )

        monkeypatch.setenv("PYLATE_DISABLE_LIK", "1")
        reference = colbert_scores(
            queries, documents, queries_mask, documents_mask
        )

        torch.testing.assert_close(fused, reference, rtol=1e-3, atol=1e-3)
