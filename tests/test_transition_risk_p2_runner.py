from types import SimpleNamespace
from aletheion_state_models.benchmarks.transition_risk import p2_runner


def test_evaluate_opened_accepts_verified_split_mapping(monkeypatch, tmp_path):
    writes = []
    monkeypatch.setattr(p2_runner, "P2_SEEDS", (29,))
    monkeypatch.setattr(p2_runner, "P2_ARMS", ("arm",))
    monkeypatch.setattr(
        p2_runner,
        "prediction_path",
        lambda root, split, seed, arm: tmp_path / f"{split}.jsonl",
    )
    monkeypatch.setattr(
        p2_runner, "checkpoint_path", lambda *args: tmp_path / "checkpoint.pt"
    )
    monkeypatch.setattr(
        p2_runner, "build_p2_arm", lambda *args: (object(), object(), {})
    )
    monkeypatch.setattr(
        p2_runner, "load_terminal_checkpoint", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        p2_runner,
        "evaluate_episodes",
        lambda *args, **kwargs: SimpleNamespace(records=("record",), metrics={}),
    )
    monkeypatch.setattr(
        p2_runner,
        "write_episode_records_jsonl",
        lambda path, records: writes.append((path.name, records)),
    )
    monkeypatch.setattr(p2_runner.gc, "collect", lambda: None)
    monkeypatch.setattr(p2_runner.torch.cuda, "is_available", lambda: False)
    p2_runner.evaluate_opened(
        tmp_path, tmp_path, {"test_id": ("episode",)}, device="cpu"
    )
    assert writes == [("test_id.jsonl", ("record",))]
