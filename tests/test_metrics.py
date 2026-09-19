import math

import pytest

from evaljev import metrics as M


def test_accuracy_and_wilson():
    assert M.accuracy([True, True, False, True]) == 0.75
    lo, hi = M.wilson(75, 100)
    # Reference: Wilson 95% interval for 75/100 is about (0.657, 0.825).
    assert lo == pytest.approx(0.6569, abs=1e-3)
    assert hi == pytest.approx(0.8245, abs=1e-3)


def test_macro_f1():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    # a: tp1 fp0 fn1 -> 2/3; b: tp2 fp1 fn0 -> 4/5; mean = 0.7333
    assert M.macro_f1(gold, pred) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_ece_perfect_and_hand_computed():
    assert M.ece([0.8] * 5, [True] * 4 + [False]) == pytest.approx(0.0)
    # Bin 0.9: conf .95 x2, one correct -> |0.5-0.95|=0.45, weight 2/4.
    # Bin 0.3: conf .35 x2, both wrong -> |0-0.35|=0.35, weight 2/4. ECE = 0.40.
    conf = [0.95, 0.95, 0.35, 0.35]
    correct = [True, False, False, False]
    assert M.ece(conf, correct) == pytest.approx(0.40)


def test_ece_weights_by_bin_size():
    # One big well-calibrated bin, one tiny badly calibrated bin: weighting must shrink the tiny one.
    conf = [0.55] * 9 + [0.95]
    correct = [True] * 5 + [False] * 4 + [False]
    # bin .5: acc 5/9=.5556 vs .55 -> .005556 * 9/10 = .005; bin .9: .95 * 1/10 = .095
    assert M.ece(conf, correct) == pytest.approx(0.005 + 0.095)


def test_brier_and_auroc():
    assert M.brier([1.0, 0.0], [True, False]) == 0.0
    assert M.brier([0.5, 0.5], [True, False]) == 0.25
    assert M.auroc([0.9, 0.8, 0.3], [True, True, False]) == 1.0
    assert M.auroc([0.5, 0.5], [True, False]) == 0.5
    assert math.isnan(M.auroc([0.9], [True]))


def test_selective():
    out = M.selective([0.95, 0.85, 0.6, 0.4], [True, False, True, False], thresholds=(0.5, 0.9))
    assert out[0] == (0.5, 0.75, pytest.approx(2 / 3))
    assert out[1] == (0.9, 0.25, 1.0)


def test_ordinal_and_agreement():
    lv = ["lo", "mid", "hi"]
    assert M.ordinal_mae(["lo", "hi"], ["mid", "lo"], lv) == 1.5
    assert M.agreement({"1": "a", "2": "b", "3": "c"}, {"1": "a", "2": "x"}) == 0.5


def test_check_accepts_the_shipped_tasks():
    from evaljev.data import load, tasks
    for t in tasks():
        rows, names = load(t)
        assert len(rows) == 300 and set(r["label"] for r in rows) <= set(names)


def test_check_reports_each_problem():
    from evaljev.data import check
    ok = [{"id": "a", "text": "hi", "label": "x"}, {"id": "b", "text": "yo", "label": "y"}]
    spec = {"kind": "choice", "instructions": "Which?"}
    assert check(spec, ok, None) == [] and check(spec, ok, ["x", "y"]) == []
    assert "kind" in check({"instructions": "q"}, ok, None)[0]
    assert "instructions" in check({"kind": "choice"}, ok, None)[0]
    assert "criteria" in check({"kind": "noul", "instructions": "q"}, ok, ["x", "y"])[0]
    assert "needs labels.json" in check({"kind": "score", "instructions": "q"}, ok, None)[0]
    assert "exactly 2" in check({"kind": "noul", "instructions": "q", "criteria": {"true": "", "false": ""}}, ok, ["x", "y", "z"])[0]
    assert "not in labels.json" in check(spec, ok, ["x"])[0]
    assert "duplicate ids" in check(spec, ok + [ok[0]], None)[0]
    assert "line 2" in check(spec, [ok[0], {"id": "c", "text": "", "label": "x"}], None)[0]
    assert check(spec, [], None) == ["no rows"]
