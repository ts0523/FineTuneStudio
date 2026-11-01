"""P1-9/10/12/14 + P2-13/17：holdout、session、指标协议、停止标记、合并终态。"""
from __future__ import annotations

import io
import json
import time

from studio.core import autotrain as at, fsutil, jobs, store as store_mod


def _sample(q: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def test_holdout_is_fixed_and_excluded_from_every_round(tmp_path):
    samples = [_sample(f"问题{i}是什么？", f"答案{i}。") for i in range(10)]
    holdout, train, hid = at.make_holdout(samples, 3, seed=7)
    assert len(holdout) == 3
    assert len(hid) == 16
    assert len(train) == 7
    keys = {at.sample_key(s) for s in holdout}
    assert all(at.sample_key(s) not in keys for s in train)

    # 下一轮即使 AI 又生成同题（答案不同），也会被 filter_holdout 剔除
    leaked_q = holdout[0]["messages"][0]["content"]
    new = [_sample(leaked_q, "另一个答案"), _sample("新问题", "新答案")]
    filtered = at.filter_holdout(new, holdout)
    assert len(filtered) == 1
    assert filtered == [_sample("新问题", "新答案")]


def test_holdout_leak_detection(tmp_path):
    holdout = [_sample("敏感题", "标准答案")]
    ds_path = tmp_path / "data.jsonl"
    lines = [json.dumps(s, ensure_ascii=False) for s in holdout]
    ds_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    leaks = at.holdout_leaks(ds_path, holdout)
    assert leaks == ["敏感题"]


def test_history_keeps_every_round(tmp_path):
    p = tmp_path / "hist.json"
    base = {"session": "s1", "holdout_id": "h1", "dataset_version": "v1"}
    at.append_history(p, base, {"round": 1, "score": 50})
    at.append_history(p, base, {"round": 2, "score": 70})
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert [r["round"] for r in obj["rounds"]] == [1, 2]
    assert obj["holdout_id"] == "h1"
    assert obj["dataset_version"] == "v1"


def test_run_metrics_protocol_matches_train_worker():
    evt = {
        "type": "done", "status": "done", "output": "out",
        "total_seconds": 123.4,
        "metrics": {"final_loss": 0.42, "best_eval_loss": 0.5, "tokens": 100,
                    "tokens_per_sec": 12.5, "steps_done": 7},
    }
    fields = at.run_fields_from_done(evt)
    assert fields["final_loss"] == 0.42
    assert fields["seconds"] == 123.4           # 必须从 total_seconds 读
    assert fields["steps_done"] == 7
    assert fields["status"] == "done"


def test_run_metrics_missing_metrics_is_not_success():
    fields = at.run_fields_from_done({"type": "done", "status": "done"})
    assert fields["final_loss"] is None
    assert fields["seconds"] is None


def test_session_guard():
    assert at.session_ok("sid-1", "sid-1")
    assert not at.session_ok("sid-1", "sid-2")
    assert not at.session_ok("", "")


def test_merge_outcome_never_green_when_verify_failed():
    assert at.merge_outcome(False, None, 0) == "written"
    assert at.merge_outcome(True, True, 0) == "verified"
    assert at.merge_outcome(True, False, 0) == "failed"
    assert at.merge_outcome(True, None, 0) == "failed"
    assert at.merge_outcome(False, None, 5) == "failed"


def test_stale_stop_flag_removed(tmp_path):
    flag = tmp_path / "studio_stop.flag"
    flag.write_text("stop old", encoding="utf-8")
    assert fsutil.prepare_stop_flag(flag) is True
    assert not flag.exists()
    assert fsutil.prepare_stop_flag(flag) is False


def test_leftover_running_marked_interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    from studio.core import paths

    paths.set_root(tmp_path)
    st = store_mod.Store()
    st.add_run({"id": "old", "name": "old", "status": "running"})
    st.mark_interrupted()
    assert st.get_run("old")["status"] == "interrupted"
    # 索引落盘后可重新读取
    st2 = store_mod.Store()
    assert st2.get_run("old")["status"] == "interrupted"


def test_job_events_carry_unique_token(tmp_path):
    """旧任务的回调必须能被 token 识别并丢弃（P1-10）。"""
    q = __import__("queue").Queue()
    job = jobs.Job("k", [], tmp_path, {}, q)
    stream = io.BytesIO(b'@@EVT@@{"type":"step","step":1}\nplain log\n')
    job._reader(stream, "out")
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except Exception:  # noqa: BLE001
            break
    evt = next(x for x in events if x[1] == "evt")[2]
    assert evt["_job"] == job.token
    job2 = jobs.Job("k", [], tmp_path, {}, __import__("queue").Queue())
    assert job2.token != job.token


def test_merge_worker_nonzero_when_verify_failed():
    import merge_worker

    assert merge_worker.final_status(True, True) == ("done", 0)
    assert merge_worker.final_status(True, False) == ("failed", 5)
    assert merge_worker.final_status(True, None) == ("failed", 5)
    assert merge_worker.final_status(False, None) == ("done", 0)


def test_data_version_changes_after_append(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(_sample("q", "a"), ensure_ascii=False) + "\n", encoding="utf-8")
    v1 = at.data_version(p)
    time.sleep(0.01)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_sample("q2", "a2"), ensure_ascii=False) + "\n")
    v2 = at.data_version(p)
    assert v1 != v2
