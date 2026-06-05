"""Tests for metadata-only prefix KV-cache trace calibration and replay."""

from __future__ import annotations

import json

import pytest

from randomize_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    baseline_lru_blocks,
)
from randomize_evolve.problems.prefix_kv_cache.trace_replay import (
    calibrate_anonymized_trace,
    load_anonymized_trace,
)
from randomize_evolve.problems.prefix_kv_cache.runner import replay_trace_report


def test_trace_loader_hides_content_and_preserves_opaque_prefix_reuse(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(
        trace_path,
        [
            _record(
                timestamp_ms=0,
                prefix_path=["root", "branch-a"],
                prompt_length=8,
                request_type="chat",
            ),
            _record(
                timestamp_ms=20,
                prefix_path=["root", "branch-a"],
                prompt_length=8,
                request_type="chat",
            ),
            _record(
                timestamp_ms=250,
                prefix_path=["root", "branch-b", "tail"],
                prompt_length=9,
                request_type="agent",
            ),
        ],
    )

    requests = load_anonymized_trace(
        trace_path,
        block_size_tokens=4,
        arrival_bucket_ms=100,
    )

    assert [request.arrival_step for request in requests] == [0, 0, 2]
    assert all(request.info.prompt_tokens == () for request in requests)
    assert requests[0].prompt_tokens == requests[1].prompt_tokens
    assert requests[0].prompt_tokens[:4] == requests[2].prompt_tokens[:4]
    assert requests[0].prompt_tokens[4:] != requests[2].prompt_tokens[4:8]

    result = PrefixKVCacheEvaluator(
        EvaluatorConfig(capacity_blocks=4, capacity_sweep_blocks=(4,))
    ).evaluate_requests(baseline_lru_blocks, requests)
    assert result.split_metrics["validation"]["token_hit_rate"] > 0.0


def test_trace_calibration_reports_mix_depth_bursts_and_lengths(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(
        trace_path,
        [
            _record(
                timestamp_ms=0,
                prefix_path=["root", "a"],
                prompt_length=8,
                output_length=32,
                request_type="chat",
            ),
            _record(
                timestamp_ms=20,
                prefix_path=["root", "b"],
                prompt_length=7,
                output_length=64,
                request_type="chat",
            ),
            _record(
                timestamp_ms=250,
                prefix_path=["root", "agent", "tail"],
                prompt_length=9,
                output_length=256,
                request_type="agent",
            ),
        ],
    )

    calibration = calibrate_anonymized_trace(trace_path, arrival_bucket_ms=100)

    assert calibration["request_count"] == 3
    assert calibration["workload_mix"]["chat"]["count"] == 2
    assert calibration["prefix_depth_blocks"]["p95"] == 3.0
    assert calibration["output_length_tokens"]["p99"] == 256.0
    assert calibration["arrival_bursts"][
        "same_bucket_request_fraction"
    ] == pytest.approx(2 / 3)


def test_trace_loader_rejects_raw_prompt_content_even_when_nested(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    record = _record(
        timestamp_ms=0,
        prefix_path=["root"],
        prompt_length=4,
        request_type="chat",
    )
    record["metadata"] = {"messages": [{"content": "secret prompt"}]}
    _write_trace(trace_path, [record])

    with pytest.raises(ValueError, match="raw-content fields are forbidden"):
        load_anonymized_trace(trace_path, block_size_tokens=4)


def test_trace_loader_rejects_prefix_depth_inconsistent_with_block_size(
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(
        trace_path,
        [
            _record(
                timestamp_ms=0,
                prefix_path=["root"],
                prompt_length=8,
                request_type="chat",
            )
        ],
    )

    with pytest.raises(ValueError, match="prefix_path depth"):
        load_anonymized_trace(trace_path, block_size_tokens=4)


def test_trace_loader_rejects_negative_priority(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    record = _record(
        timestamp_ms=0,
        prefix_path=["root"],
        prompt_length=4,
        request_type="chat",
    )
    record["priority"] = -1
    _write_trace(trace_path, [record])

    with pytest.raises(ValueError, match="priority must be an integer >= 0"):
        load_anonymized_trace(trace_path, block_size_tokens=4)


def test_replay_report_runs_deployable_baselines_on_fixed_trace(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "replay.json"
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(
        "from randomize_evolve.evaluators.prefix_kv_cache import baseline_lru_blocks\n"
        "\n"
        "def build_candidate(capacity_blocks, block_size_tokens, seed=None):\n"
        "    return baseline_lru_blocks(capacity_blocks, block_size_tokens, seed)\n",
        encoding="utf-8",
    )
    _write_trace(
        trace_path,
        [
            _record(
                timestamp_ms=index * 100,
                prefix_path=["root", "branch"],
                prompt_length=8,
                request_type="chat",
            )
            for index in range(3)
        ],
    )

    payload = replay_trace_report(
        trace_path,
        output_path=output_path,
        candidate_program=candidate_path,
        arrival_bucket_ms=100,
        request_limit=None,
        config_file="configs/prefix_kv_cache.yaml",
        capacity_sweep_blocks=(4,),
        block_size_tokens=4,
    )

    assert output_path.exists()
    assert payload["request_count"] == 3
    assert payload["capacity_blocks"] == [4]
    assert payload["results"]["candidate"]["success"] is True
    assert (
        payload["results"]["lru"]["split_metrics"]["validation"]["token_hit_rate"] > 0.0
    )


def _record(
    *,
    timestamp_ms: int,
    prefix_path: list[str],
    prompt_length: int,
    request_type: str,
    output_length: int = 64,
) -> dict:
    return {
        "timestamp_ms": timestamp_ms,
        "tenant_hash": "tenant-a",
        "session_hash": "session-a",
        "request_type": request_type,
        "priority": 0,
        "prompt_length": prompt_length,
        "output_length": output_length,
        "predicted_output_length": output_length,
        "prefix_path": prefix_path,
    }


def _write_trace(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
