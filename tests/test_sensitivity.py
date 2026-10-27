"""Tests for scripts/sensitivity_analysis.py (offline threshold sensitivity, no GPU, no torch)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import sensitivity_analysis as sa  # noqa: E402

FPS = 30.0
N_FRAMES = 300
PAIRS = ["0_1", "0_2"]


def _synthetic_frame_table() -> pd.DataFrame:
    """Two pairs x 300 frames (interleaved per frame, like the real table).

    Pair 0_1: score_n2n 0.55 on frames 50-89 (40 frames) and 0.45 on frames 90-99 (10 more).
    Pair 0_2: score_n2ag 0.8 on frames 150-164 (15 frames = 0.5 s).
    Every other score is 0.
    """
    rows = []
    for f in range(N_FRAMES):
        for pk in PAIRS:
            r = {"frame_idx": f, "pair_key": pk, "score_n2n": 0.0, "score_n2ag": 0.0,
                 "score_fol": 0.0, "score_sbs": 0.0, "score_n2b": 0.0}
            if pk == "0_1" and 50 <= f < 90:
                r["score_n2n"] = 0.55
            elif pk == "0_1" and 90 <= f < 100:
                r["score_n2n"] = 0.45
            if pk == "0_2" and 150 <= f < 165:
                r["score_n2ag"] = 0.8
            rows.append(r)
    df = pd.DataFrame(rows)
    scores = df[list(sa.SCORE_COLUMNS)].to_numpy()
    df["contact_type"] = sa.raw_labels(scores, 0.5, 0.35)
    return df


@pytest.fixture(scope="module")
def table() -> sa.FrameTable:
    return sa.prepare_table(_synthetic_frame_table())


@pytest.fixture(scope="module")
def baseline() -> dict:
    return sa.load_baseline()


def _stats(table, baseline, **override):
    params = dict(baseline)
    params.update(override)
    return sa.evaluate(table, params, FPS)


def test_baseline_values_come_from_the_postprocess_yaml(baseline):
    assert baseline["activation_threshold"] == 0.5
    assert baseline["activation_threshold_rare"] == 0.35
    assert baseline["smoothing_window"] == 5
    assert baseline["gap_bridging_max_gap"] == 3
    assert baseline["min_bout_sec"] == pytest.approx(0.3)


def test_activation_threshold_changes_event_length(table, baseline):
    at_05 = _stats(table, baseline, activation_threshold=0.5)["N2N"]
    at_04 = _stats(table, baseline, activation_threshold=0.4)["N2N"]
    assert at_05["n_events"] == 1 and at_04["n_events"] == 1
    assert at_05["total_sec"] == pytest.approx(40 / FPS, abs=1e-3)   # only the 0.55 frames
    assert at_04["total_sec"] == pytest.approx(50 / FPS, abs=1e-3)   # the 0.45 frames join the event
    assert at_04["mean_sec"] > at_05["mean_sec"]
    assert at_05["n_pairs"] == 1 and at_05["n_pair_frames"] == 40


def test_min_bout_one_second_removes_a_half_second_event(table, baseline):
    kept = _stats(table, baseline, min_bout_sec=0.3)
    dropped = _stats(table, baseline, min_bout_sec=1.0)
    assert kept["N2AG"]["n_events"] == 1
    assert kept["N2AG"]["total_sec"] == pytest.approx(0.5, abs=1e-3)
    assert dropped["N2AG"]["n_events"] == 0
    assert np.isnan(dropped["N2AG"]["mean_sec"])
    assert dropped["N2N"]["n_events"] == 1   # the 1.33 s N2N event survives a 1 s floor


def test_long_table_has_every_parameter_value_and_type(table, baseline):
    grid = sa.complete_grid({k: list(v) for k, v in sa.DEFAULT_GRID.items()}, baseline)
    long_df = sa.run_sensitivity(table, FPS, grid, baseline)
    assert list(long_df.columns[:6]) == ["parameter", "value", "contact_type", "n_events", "total_sec", "mean_sec"]
    assert set(long_df["parameter"]) == set(sa.PARAMETERS)
    for param, values in grid.items():
        sub = long_df[long_df["parameter"] == param]
        assert sorted(float(v) for v in sub["value"].unique()) == pytest.approx(sorted(values))
        assert set(sub["contact_type"]) == set(sa.TYPE_ORDER)
        base_rows = sub[sub["is_baseline"]]
        assert len(base_rows) == len(sa.TYPE_ORDER)
        assert float(base_rows["value"].iloc[0]) == pytest.approx(baseline[param])
    # The baseline row of every parameter is the same result (one-at-a-time design).
    base = long_df[long_df["is_baseline"]].groupby("contact_type")["n_events"].nunique()
    assert (base == 1).all()


def test_baseline_value_is_added_to_a_grid_that_lacks_it(baseline):
    grid = sa.complete_grid({"smoothing_window": [7], "min_bout_sec": [1.0, 0.5]}, baseline)
    assert grid["smoothing_window"] == [5, 7]
    assert grid["min_bout_sec"] == [0.3, 0.5, 1.0]


def test_write_outputs_csv_markdown_and_pngs(table, baseline, tmp_path):
    grid = sa.complete_grid({k: list(v) for k, v in sa.DEFAULT_GRID.items()}, baseline)
    long_df = sa.run_sensitivity(table, FPS, grid, baseline)
    info = {"run": "synthetic", "fps": FPS, "n_rows": table.n_rows, "n_pairs": 2, "n_frames": N_FRAMES,
            "baseline": baseline, "agreement": 1.0}
    written = sa.write_outputs(long_df, info, tmp_path / "out")
    names = {p.name for p in written}
    assert "sensitivity_long.csv" in names and "sensitivity_summary.md" in names
    for param in sa.PARAMETERS:
        png = tmp_path / "out" / f"sensitivity_{param}.png"
        assert png.exists() and png.stat().st_size > 1000
    back = pd.read_csv(tmp_path / "out" / "sensitivity_long.csv")
    assert set(back["parameter"]) == set(sa.PARAMETERS)
    assert len(back) == sum(len(v) for v in grid.values()) * len(sa.TYPE_ORDER)
    md = (tmp_path / "out" / "sensitivity_summary.md").read_text(encoding="utf-8")
    assert "(baseline)" in md and "cannot be re-evaluated offline" in md and "Schmitt" in md
    assert "Robustness" in md
    statement = sa.robustness_statement(long_df)
    assert statement.count("N2N") == 2   # once per grid (activation, minimum duration)


def test_vectorised_labels_match_scoremap():
    rng = np.random.default_rng(1)
    levels = np.array([0.0, 0.2, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0])   # lots of ties and exact thresholds
    scores = levels[rng.integers(0, len(levels), size=(3000, 5))]
    for act in (0.4, 0.5, 0.7):
        for rare in (0.25, 0.35, 0.45):
            assert (sa.raw_labels(scores, act, rare) == sa.reference_labels(scores, act, rare)).all()


def test_missing_scores_count_as_zero():
    pf = _synthetic_frame_table().drop(columns=["score_fol"])
    pf.loc[pf.index[:50], "score_n2n"] = np.nan
    t = sa.prepare_table(pf)
    assert t.scores.shape == (len(pf), 5)
    assert not np.isnan(t.scores).any()
    assert (t.scores[:, sa.SCORE_COLUMNS.index("score_fol")] == 0).all()
    with pytest.raises(ValueError):
        sa.prepare_table(pf.drop(columns=[c for c in sa.SCORE_COLUMNS if c in pf.columns]))


def test_cli_end_to_end_with_overrides(tmp_path, capsys):
    run = tmp_path / "run"
    (run / "contacts").mkdir(parents=True)
    _synthetic_frame_table().to_csv(run / "contacts" / "contacts_per_frame.csv", index=False)
    out = tmp_path / "sens"
    rc = sa.main([str(run), "--out", str(out), "min_bout_sec=[0.3,1.0]", "smoothing_window=7",
                  "activation_threshold=[0.4,0.5]", "fps=30"])
    assert rc == 0
    long_df = pd.read_csv(out / "sensitivity_long.csv")
    assert sorted(long_df[long_df["parameter"] == "smoothing_window"]["value"].unique()) == [5, 7]
    assert sorted(long_df[long_df["parameter"] == "min_bout_sec"]["value"].unique()) == [0.3, 1.0]
    n2ag = long_df[(long_df["parameter"] == "min_bout_sec") & (long_df["contact_type"] == "N2AG")]
    assert n2ag.set_index("value")["n_events"].to_dict() == {0.3: 1, 1.0: 0}
    printed = capsys.readouterr().out
    assert "Robustness." in printed
    assert sorted(long_df[long_df["parameter"] == "activation_threshold"]["value"].unique()) == [0.4, 0.5]
    assert (out / "sensitivity_summary.md").exists()
