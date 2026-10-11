"""BFCL FC 路徑的格式錯誤處理、推理內容保存，以及 --download-dataset bfcl 的單元測試。"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from twinkle_eval.metrics import create_metric_pair
from twinkle_eval.runners.evaluator import Evaluator

_FUNCTIONS = [
    {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "dict",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }
]


def _row(i: int) -> dict:
    return {
        "id": f"simple_{i}",
        "question": json.dumps([{"role": "user", "content": f"Q{i}: add 1 and 2"}]),
        "functions": json.dumps(_FUNCTIONS),
        "answer": json.dumps(
            {"category": "simple", "ground_truth": [{"add": {"a": [1], "b": [2]}}]}
        ),
    }


def _completion(arguments: str, reasoning=None) -> SimpleNamespace:
    tool_call = SimpleNamespace(function=SimpleNamespace(name="add", arguments=arguments))
    message = SimpleNamespace(
        content=None, tool_calls=[tool_call], reasoning=reasoning, reasoning_content=None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


def _run(tmp_path, monkeypatch, responses):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "simple.jsonl"
    data.write_text("\n".join(json.dumps(_row(i)) for i in range(len(responses))))

    llm = MagicMock()
    llm.call.side_effect = lambda *args, **kwargs: responses[
        int(args[7][0]["content"].split(":")[0][1:])
    ]
    extractor, scorer = create_metric_pair("bfcl_fc", {})
    evaluator = Evaluator(
        llm=llm,
        extractor=extractor,
        scorer=scorer,
        config={"llm_api": {"api_rate_limit": -1}, "evaluation": {}},
        eval_method="bfcl_fc",
    )
    _, metrics, path = evaluator.evaluate_file(str(data), "ts_run0")
    with open(path, encoding="utf-8") as f:
        rows = sorted((json.loads(line) for line in f), key=lambda r: r["question_id"])
    return metrics, rows


def test_malformed_arguments_do_not_abort_the_file(tmp_path, monkeypatch):
    """一題的 tool call 參數不是合法 JSON，不得中止整個檔案、遺失其他題目。"""
    metrics, rows = _run(
        tmp_path,
        monkeypatch,
        [
            _completion('{"a": 1, "b": 2}'),
            _completion('{"a": 1, "b":'),  # 被截斷的 JSON
            _completion('{"a": 1, "b": 2}'),
        ],
    )

    assert metrics["total_count"] == 3
    assert metrics["unparsed_count"] == 1
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert [r["is_correct"] for r in rows] == [True, False, True]
    # 原始參數字串保留在 JSONL，方便除錯
    assert rows[1]["predicted_answer"] is None
    assert rows[1]["tool_calls"] == [{"name": "add", "arguments": '{"a": 1, "b":'}]


def test_empty_arguments_are_treated_as_no_arguments(tmp_path, monkeypatch):
    _, rows = _run(tmp_path, monkeypatch, [_completion("")])
    assert rows[0]["predicted_answer"] is not None


def test_reasoning_is_saved(tmp_path, monkeypatch):
    _, rows = _run(tmp_path, monkeypatch, [_completion('{"a": 1, "b": 2}', reasoning="1+2")])
    assert rows[0]["llm_reasoning_output"] == "1+2"


def test_download_bfcl_merges_raw_files(tmp_path, monkeypatch):
    """--download-dataset bfcl 逐檔下載原始題目與答案，合併為 {category}.jsonl。"""
    import huggingface_hub

    from twinkle_eval.benchmarks import BENCHMARK_REGISTRY, download_benchmarks

    raw = tmp_path / "raw"
    (raw / "possible_answer").mkdir(parents=True)
    for category in BENCHMARK_REGISTRY["bfcl"]["categories"]:
        name = f"BFCL_v3_{category}.json"
        (raw / name).write_text(
            json.dumps(
                {
                    "id": f"{category}_0",
                    "question": [[{"role": "user", "content": "hi"}]],
                    "function": _FUNCTIONS,
                }
            )
        )
        (raw / "possible_answer" / name).write_text(
            json.dumps({"id": f"{category}_0", "ground_truth": [{"add": {"a": [1]}}]})
        )

    requested = []

    def fake_download(repo_id, filename, repo_type=None, **kwargs):
        requested.append((repo_id, filename, repo_type))
        return str(raw / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert download_benchmarks(["bfcl"], str(tmp_path / "datasets")) == 0

    out = tmp_path / "datasets" / "bfcl"
    categories = BENCHMARK_REGISTRY["bfcl"]["categories"]
    assert sorted(os.listdir(out)) == sorted(f"{c}.jsonl" for c in categories)
    row = json.loads((out / "simple.jsonl").read_text())
    assert json.loads(row["answer"])["category"] == "simple"
    assert all(
        r == ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", r[1], "dataset")
        for r in requested
    )
    assert len(requested) == 2 * len(categories)
