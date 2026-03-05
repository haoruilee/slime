import copy
import logging

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_processor, load_tokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False
EVAL_DATASETS = {}


def _ensure_initialized(args):
    global TOKENIZER, PROCESSOR, MASK_GENERATOR
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Rollout function for SFT training and validation.

    When ``evaluation=False`` (training mode): tokenizes samples from
    ``data_buffer`` and attaches tokens, loss masks, and response lengths to
    each sample.

    When ``evaluation=True`` (validation mode): iterates over
    ``args.eval_datasets``, tokenizes every sample in each dataset, and
    returns a :class:`~slime.rollout.base_types.RolloutFnEvalOutput` whose
    ``data`` field maps dataset name → ``{"samples": [tokenized_sample, ...]}``.
    The caller (``RolloutManager.generate_sft_val_data``) converts those
    samples into Megatron-compatible rollout data refs so that the training
    actor can run a forward-only pass to compute validation NLL loss.

    Args:
        args: The whole args namespace.
        rollout_id: Integer rollout id (used for deterministic data selection).
        data_buffer: Training data source (used only when ``evaluation=False``).
        evaluation: Whether this call is for validation rather than training.

    Returns:
        :class:`~slime.rollout.base_types.RolloutFnTrainOutput` when
        ``evaluation=False``, or
        :class:`~slime.rollout.base_types.RolloutFnEvalOutput` when
        ``evaluation=True``.
    """
    global SAMPLE_PRINTED

    _ensure_initialized(args)

    if evaluation:
        return _generate_eval_rollout(args)

    assert args.rollout_global_dataset

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        tools = sample.metadata.get("tools", None) if sample.metadata else None

        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)

        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"sft_rollout::generate_rollout example data: {sample=} (raw){messages=} "
                f"(raw){token_ids=} (raw){loss_mask=} {response_length=}"
            )
            SAMPLE_PRINTED = True

    return RolloutFnTrainOutput(samples=samples)


def _generate_eval_rollout(args) -> RolloutFnEvalOutput:
    """Tokenize every sample in each configured eval dataset for SFT validation.

    Builds a :class:`~slime.rollout.base_types.RolloutFnEvalOutput` where
    ``data`` maps each dataset name to a dict containing a ``"samples"`` list
    of fully tokenized :class:`~slime.utils.types.Sample` objects.  Each
    sample has ``tokens``, ``loss_mask``, ``response_length``, and a dummy
    ``reward=0`` (rewards are not used for SFT validation loss).

    This output is consumed by ``RolloutManager.generate_sft_val_data``, which
    converts the samples into Megatron-compatible rollout data refs and passes
    them to the training actor for a forward-only NLL computation.
    """
    from slime.utils.data import Dataset

    eval_datasets = getattr(args, "eval_datasets", []) or []
    if not eval_datasets:
        raise ValueError(
            "No eval datasets configured for SFT validation. "
            "Use --eval-prompt-data or --eval-config to specify a validation dataset "
            "together with --eval-interval."
        )

    data = {}
    for dataset_cfg in eval_datasets:
        cache_key = dataset_cfg.path
        if cache_key not in EVAL_DATASETS:
            max_len = getattr(args, "eval_max_prompt_len", None) or args.rollout_max_prompt_len
            EVAL_DATASETS[cache_key] = Dataset(
                path=dataset_cfg.path,
                tokenizer=TOKENIZER,
                processor=PROCESSOR,
                max_length=max_len,
                prompt_key=dataset_cfg.input_key,
                label_key=getattr(dataset_cfg, "label_key", None),
                metadata_key=getattr(dataset_cfg, "metadata_key", None),
                tool_key=getattr(dataset_cfg, "tool_key", None),
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            )
        dataset = EVAL_DATASETS[cache_key]

        samples = []
        for prompt_sample in dataset.samples:
            sample = copy.deepcopy(prompt_sample)
            messages = sample.prompt
            tools = sample.metadata.get("tools", None) if sample.metadata else None

            token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
            response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

            sample.tokens = token_ids
            sample.response_length = response_length
            sample.reward = 0
            sample.loss_mask = loss_mask[-response_length:]
            samples.append(sample)

        logger.info(
            f"sft_rollout::_generate_eval_rollout: dataset '{dataset_cfg.name}': {len(samples)} samples tokenized"
        )
        data[dataset_cfg.name] = {"samples": samples}

    return RolloutFnEvalOutput(data=data)
