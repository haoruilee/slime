"""CI test: SFT training with validation-loss monitoring.

Trains Qwen2.5-0.5B for a few rollouts in SFT mode (--debug-train-only) and
verifies that validation NLL loss is computed and logged at each eval interval.
"""

import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ "

    sft_args = (
        "--rollout-function-path slime.rollout.sft_rollout.generate_rollout "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 3 "
        "--rollout-batch-size 32 "
        "--global-batch-size 32 "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        "--debug-train-only "
    )

    eval_args = (
        "--eval-prompt-data gsm8k_val /root/datasets/gsm8k/test.parquet "
        "--eval-interval 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style cosine "
        "--min-lr 1e-6 "
        "--lr-warmup-fraction 0.1 "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.95 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{sft_args} "
        f"{eval_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    execute()
