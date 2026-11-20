import csv
import json
import os
import re
import pytest
from analysis import trackio_logs as tl

"""
analysis/trackio_logs.py: training and beam search logs -> CSV files in the trackio import format (step column,
optional ISO timestamp, numeric metrics with names trackio keeps unchanged).
"""

TRAIN_LOG = """Using cpu device
Num timesteps: 800
Best mean reward: -inf - Last mean reward per episode: 7.02
New best mean reward: 7.02
Saving new best model to tmp_Tainan/best_model_Tainan_800
Num timesteps: 1200
Best mean reward: 7.02 - Last mean reward per episode: 6.90
Num timesteps: 1600
Best mean reward: 7.02 - Last mean reward per episode: 7.10
New best mean reward: 7.10
Saving new best model to tmp_Tainan/best_model_Tainan_1600
"""

BEAM_LOG = """start: score 0.2333, Q [0.464 0.468 0.489 0.478 0.468]
existing plan, step 0: 2 stations, avg travel time 3.861 min, avg waiting time 3.030 min
step 25 beam 4 best live 0.2333 (key 0.7575, 0 stations, budget 4981250) best done - (14s) backbone live
  current plan of the best member, step 25: 2 stations, avg travel time 3.500 min, avg waiting time 2.000 min
step 50 beam 2 best live 0.9000 (key 1.2000, 40 stations, budget 4000000) best done 1.0000 (30s)
step 52 beam 0 best live nan (key nan, 0 stations, budget 0) best done 1.1000 (33s)
{'city': 'Synth', 'existing_score': 0.23333333333333334, 'score': 1.1, 'improvement': 0.8666666666666667}
peak memory of coordinator + beam members: 0.5 GB (PSS)
"""


def write(path, text):
    path.write_text(text)
    return str(path)


def read(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def trackio_name(column):
    """column name as trackio.import_csv stores it (utils.simplify_column_names)"""
    return re.sub(r"[^a-zA-Z0-9/]", "", column)[:10]


@pytest.mark.parametrize("columns", [tl.ORIGINAL_COLUMNS, tl.SOURCE_COLUMNS, tl.BEAM_COLUMNS])
def test_column_names_survive_trackio(columns):
    assert columns[0] == "step"
    assert [trackio_name(column) for column in columns] == columns


def test_epsilon_schedule():
    assert tl.epsilon(0, 20000) == 1.0
    assert tl.epsilon(1000, 20000) == pytest.approx(0.525)
    assert tl.epsilon(2000, 20000) == pytest.approx(0.05)
    assert tl.epsilon(15000, 20000) == pytest.approx(0.05)


def test_original_training(tmp_path):
    monitor = write(tmp_path / "monitor.csv", '#{"t_start": 1792136651.3, "env_id": null}\nr,l,t\n7.909102,739,6687.5\n')
    log = write(tmp_path / "original_model.log", "original model: existing 0.2333 best 8.1424 improvement 7.9091 "
                                                 "stations 755 steps 739 seconds 3957\n")
    rows = tl.original_training(write(tmp_path / "train.log", TRAIN_LOG), monitor, log, total=20000)
    assert [row["step"] for row in rows] == [800, 1200, 1600]
    assert rows[0] == {"step": 800, "saved": 1, "epsilon": 0.62, "reward/avg": 7.02, "reward/max": 7.02}
    assert (rows[1]["reward/avg"], rows[1]["reward/max"], rows[1]["saved"]) == (6.9, 7.02, 0)
    assert (rows[2]["reward/max"], rows[2]["eval/imp"], rows[2]["eval/stns"]) == (7.1, 7.909102, 755)


def test_original_training_without_evaluation(tmp_path):
    rows = tl.original_training(write(tmp_path / "train.log", TRAIN_LOG))
    assert "eval/imp" not in rows[-1]


def test_source_training(tmp_path):
    monitor = write(tmp_path / "monitor.csv", '#{"t_start": 1793693631.5, "env_id": null}\n'
                                              'r,l,t\n7.0,600,1800.0\n6.5,500,3600.0\n')
    selection = write(tmp_path / "selection.csv", "train_steps,improvement,n_stations\n1000,8.4,395\n600,1.5,20\n")
    config = write(tmp_path / "config.json", json.dumps({"steps": 10000}))
    rows = tl.source_training(monitor, selection, config)
    assert [row["step"] for row in rows] == [600, 1000, 1100]
    assert rows[0]["timestamp"] == "2026-11-03T08:43:51.500000+00:00"
    assert (rows[0]["hours"], rows[0]["reward"], rows[0]["length"], rows[0]["eval/imp"]) == (0.5, 7.0, 600, 1.5)
    assert rows[0]["epsilon"] == pytest.approx(1 - 0.95 * 600 / 1000)
    assert rows[1] == {"step": 1000, "eval/imp": 8.4, "eval/stns": 395}
    assert rows[2]["epsilon"] == 0.05


def test_beam_progress(tmp_path):
    rows = tl.beam_progress(write(tmp_path / "beam.log", BEAM_LOG))
    assert [row["step"] for row in rows] == [0, 25, 50, 52]
    assert rows[0] == {"step": 0, "score": 0.2333, "improve": 0.0, "travel": 3.861, "waiting": 3.03}
    assert rows[1]["live/best"] == 0.2333 and "done/best" not in rows[1]
    assert (rows[1]["beam"], rows[1]["stations"], rows[1]["budget"], rows[1]["travel"]) == (4, 0, 4981250, 3.5)
    assert (rows[2]["score"], rows[2]["live/best"], rows[2]["done/best"], rows[2]["seconds"]) == (1.0, 0.9, 1.0, 30)
    assert rows[3]["score"] == 1.1 and "live/best" not in rows[3] and rows[3]["beam"] == 0
    assert rows[3]["improve"] == pytest.approx(0.8667)


def test_beam_progress_without_result_line(tmp_path):
    rows = tl.beam_progress(write(tmp_path / "beam.log", BEAM_LOG.split("{")[0]))
    assert "improve" not in rows[-1]


def test_main_writes_trackio_files(tmp_path, capsys):
    root = tmp_path / "repo"
    for folder in ["tmp_Tainan", "runs/source_Tainan", "Results/Transfer", "logs/beam", "logs/training"]:
        (root / folder).mkdir(parents=True)
    write(root / "tmp_Tainan" / "train.log", TRAIN_LOG)
    write(root / "tmp_Tainan" / "monitor.csv", '#{"t_start": 1.0, "env_id": null}\nr,l,t\n7.9,739,10.0\n')
    write(root / "logs" / "training" / "original_model.log", "stations 755\n")
    write(root / "runs" / "source_Tainan" / "monitor.csv", '#{"t_start": 1.0, "env_id": null}\nr,l,t\n7.0,600,5.0\n')
    write(root / "runs" / "source_Tainan" / "config.json", json.dumps({"steps": 20000}))
    write(root / "Results" / "Transfer" / "source_selection.csv", "train_steps,improvement,n_stations\n1000,8.4,395\n")
    write(root / "logs" / "beam" / "B4k2_e1.log", BEAM_LOG)
    write(root / "logs" / "beam" / "B1k1_e1.log", BEAM_LOG)
    out = tmp_path / "trackio"
    written = tl.main(["--root", str(root), "--out", str(out), "--seeds", "1", "2"])
    assert sorted(os.path.relpath(path, out) for path in written) == [
        "finetuning/Taipei_B4k2_e1.csv", "finetuning/Taipei_zeroshot_e1.csv",
        "training/original_Tainan.csv", "training/source_Tainan.csv"]
    beam = read(out / "finetuning" / "Taipei_B4k2_e1.csv")
    assert list(beam[0]) == tl.BEAM_COLUMNS and beam[1]["done/best"] == ""
    assert read(out / "training" / "source_Tainan.csv")[1]["eval/imp"] == "8.4"
    assert "original_Tainan.csv: 3 rows" in capsys.readouterr().out


def test_empty_columns_are_left_out(tmp_path):
    rows = tl.beam_progress(write(tmp_path / "beam.log", "\n".join(l for l in BEAM_LOG.splitlines() if "avg" not in l)))
    tl.write_csv(rows, str(tmp_path / "out" / "beam.csv"), tl.BEAM_COLUMNS)
    assert list(read(tmp_path / "out" / "beam.csv")[0]) == tl.BEAM_COLUMNS[:-2]


def test_number():
    assert tl.number("1.5") == 1.5
    assert tl.number("-") is None
    assert tl.number("nan") is None
