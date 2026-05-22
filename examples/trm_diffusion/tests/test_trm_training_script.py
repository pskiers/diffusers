import os
import subprocess
import shutil
import pytest

SMALL_THINKER_ARGS = [
    "thinker.hidden_size=16",
    "thinker.L_cycles=2",
    "thinker.H_cycles=1",
    "thinker.n_sup=2",
    "thinker.puzzle_emb_ndim=16",
]


SMALL_PAINTER_ARGS = [
    "painter.bridge_channels=16",
    "painter.painter_channels=[16, 16]",
    "painter.painter_layers_per_block=1",
    "painter.enc_channels=16",
    "painter.enc_hidden_channels=[16]",
]

FAST_TRAINING_ARGS = [
    "compile=false",
    "train.batch_size=2",
    "train.num_steps=3",
    "train.eval_every=2",
    "train.save_every=2",
    "train.log_every=1",
    "train.gradient_accumulation_steps=1",
    "train.eval_num_samples=2",
    "train.eval_batch_size=2",
    "train.eval_num_log_images=2",
]

TWO_STAGE_PT_ARGS = [
    "painter_stage.n_sup=2",
    "painter_stage.H_cycles=1",
    "painter_stage.L_cycles=2",
    "thinker_stage.n_sup=2",
]

RESUME_FROM_CKPT_ARG = "resume_from_checkpoint"

SCRIPT_PATH = "train_trm.py"

TMP_DIR = "tmp_test_dir"


def run_script(script_path, output_dir, args, resume_from_checkpoint=False, cleanup=True):
    try:
        os.makedirs(output_dir, exist_ok=True)
        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"
        env["WANDB_DIR"] = output_dir

        cmd = ["python", script_path, f"output_dir={output_dir}"] + args
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"Initial Run Failed!\n{res.stdout}\n{res.stderr}"

        if resume_from_checkpoint:
            cmd_resume = cmd + [f"{RESUME_FROM_CKPT_ARG}={output_dir}/checkpoint_final.pt"]
            res = subprocess.run(cmd_resume, env=env, capture_output=True, text=True)
            assert res.returncode == 0, f"Resume Run Failed!\n{res.stdout}\n{res.stderr}"
    finally:
        if cleanup:
            shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.trm
def test_original_trm():
    args = ["experiment=sudoku"] + FAST_TRAINING_ARGS + SMALL_THINKER_ARGS
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.v0tok
def test_pt_v0tok():
    args = ["experiment=v0tok"] + FAST_TRAINING_ARGS + SMALL_THINKER_ARGS + SMALL_PAINTER_ARGS
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.v0tok
@pytest.mark.two_stage_pt
def test_pt_v0tok_2stage():
    args = (
        ["experiment=v0tok_two_stage"]
        + FAST_TRAINING_ARGS
        + SMALL_THINKER_ARGS
        + SMALL_PAINTER_ARGS
        + TWO_STAGE_PT_ARGS
    )
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.v0
def test_pt_v0():
    args = ["experiment=v0"] + FAST_TRAINING_ARGS + SMALL_THINKER_ARGS + SMALL_PAINTER_ARGS
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.v1
def test_pt_v1():
    args = (
        ["experiment=v1_two_stage"] + FAST_TRAINING_ARGS + SMALL_THINKER_ARGS + SMALL_PAINTER_ARGS + TWO_STAGE_PT_ARGS
    )
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.v2
def test_pt_v2():
    args = (
        ["experiment=v2_two_stage"] + FAST_TRAINING_ARGS + SMALL_THINKER_ARGS + SMALL_PAINTER_ARGS + TWO_STAGE_PT_ARGS
    )
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.v3
def test_pt_v3():
    args = (
        ["experiment=v3_two_stage"] + FAST_TRAINING_ARGS + SMALL_THINKER_ARGS + SMALL_PAINTER_ARGS + TWO_STAGE_PT_ARGS
    )
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.v4
def test_pt_v4():
    args = (
        ["experiment=v4_two_stage"] + FAST_TRAINING_ARGS + SMALL_THINKER_ARGS + SMALL_PAINTER_ARGS + TWO_STAGE_PT_ARGS
    )
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.painter_concat
def test_painter_concat():
    args = ["experiment=standalone_painter"] + FAST_TRAINING_ARGS + SMALL_PAINTER_ARGS
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.painter_controlnet
def test_painter_controlnet():
    args = ["experiment=standalone_painter_control"] + FAST_TRAINING_ARGS + SMALL_PAINTER_ARGS
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.painter_spade
def test_painter_spade():
    args = ["experiment=standalone_painter_spade"] + FAST_TRAINING_ARGS + SMALL_PAINTER_ARGS
    run_script(
        script_path=SCRIPT_PATH,
        output_dir=TMP_DIR,
        args=args,
        resume_from_checkpoint=True,
    )


@pytest.mark.frozen_painter_v0tok
def test_thinker_frozen_painter_v0tok():
    painter_dir = TMP_DIR + "_standalone_painter"
    args = ["experiment=standalone_painter"] + FAST_TRAINING_ARGS + SMALL_PAINTER_ARGS
    run_script(script_path=SCRIPT_PATH, output_dir=painter_dir, args=args, resume_from_checkpoint=False, cleanup=False)
    args = (
        ["experiment=thinker_frozen_painter", f"painter.painter_checkpoint={painter_dir}/checkpoint_final.pt"]
        + FAST_TRAINING_ARGS
        + SMALL_PAINTER_ARGS
    )
    run_script(script_path=SCRIPT_PATH, output_dir=TMP_DIR, args=args, resume_from_checkpoint=True, cleanup=True)
    shutil.rmtree(painter_dir, ignore_errors=True)
