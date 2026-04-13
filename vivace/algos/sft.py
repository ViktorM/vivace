"""Optional SFT warmup.

Standalone supervised fine-tuning loop. For most base models on GSM8K
with the `<think>/<answer>` system prompt, SFT is NOT needed — the
model already follows the format. Keep this path optional.

When you DO use it: run SFT first, save a checkpoint, then point
`vivace-train` at that checkpoint as its starting weights. Two-stage
workflow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vivace.algos.types import SFTConfig


def build_sft_data(examples: list, n: int, tokenizer, make_prompt) -> list[dict]:
    """Materialize a list of {prompt, target} dicts from raw env examples.

    THEORY
    ------
    The SFT target is the IDEAL response in the same format the policy
    will be expected to produce: <think>{reasoning}</think><answer>{ans}</answer>.
    For GSM8K this means concatenating the dataset's `reasoning` field
    (split out from the `####`-delimited answer) with the final answer.

    HINTS
    -----
    - For each example: prompt = make_prompt(ex.problem)
    - target = f"<think>\\n{ex.metadata['reasoning']}\\n</think>\\n"
               f"<answer>\\n{ex.answer}\\n</answer>\\n{tokenizer.eos_token}"
    - return [{"prompt": ..., "target": ...}, ...] truncated to n
    """
    data = []
    for ex in examples[:n]:
        prompt = make_prompt(ex.problem)
        # Build the ideal response in the same <think>/<answer> format
        # the RL reward functions expect.
        reasoning = ex.metadata.get("reasoning", "") if ex.metadata else ""
        target = (
            f"<think>\n{reasoning}\n</think>\n"
            f"<answer>\n{ex.answer}\n</answer>\n"
            f"{tokenizer.eos_token}"
        )
        data.append({"prompt": prompt, "target": target})
    return data


def encode_sft_batch(batch: list[dict], tokenizer, device) -> dict[str, torch.Tensor]:
    """Tokenize a batch into input_ids / attention_mask / labels.

    THEORY
    ------
    Labels are -100 for prompt tokens (ignored by cross-entropy) and the
    actual token id for target tokens. This way the model only learns
    to predict the response, not to re-emit the prompt.

    Padding side: LEFT (matches the rest of the codebase). With left-padding,
    the attention mask is [0]*pad + [1]*real.

    HINTS
    -----
    - For each ex: p_ids = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
                   f_ids = tokenizer(ex["prompt"] + ex["target"], add_special_tokens=False).input_ids
                   labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    - Pad all sequences to max_len in the batch (left-pad).
    - Return tensors on `device`, dtype=torch.long.

    GOTCHAS
    -------
    - Use add_special_tokens=False both times so the prompt prefix lengths
      match exactly. Otherwise the labels mask will be off by one or two.
    - Cross-entropy ignores -100 by default — don't override ignore_index.
    """
    # TODO: implement.
    raise NotImplementedError


def sft_loss(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """Standard next-token cross-entropy with prompt tokens masked out.

    THEORY
    ------
    Shift labels by 1 (predict token t+1 from position t). Reshape to
    flat (-1, vocab) and (-1,), let CE handle the -100 ignores.

    HINTS
    -----
    - logits = model(input_ids=..., attention_mask=...).logits
    - shift_logits = logits[:, :-1, :].float()
    - shift_labels = labels[:, 1:]
    - return F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1),
                             ignore_index=-100)
    """
    # TODO: implement.
    raise NotImplementedError


def sft_train_loop(model, tokenizer, train_data: list[dict], cfg: SFTConfig, optimizer) -> None:
    """N-step SFT training loop with linear warmup.

    Standard pattern: build a DataLoader from `train_data`, iterate up to
    cfg.steps, compute sft_loss, backward, clip grads, optimizer.step,
    scheduler.step. No tricks.

    HINTS
    -----
    - from torch.utils.data import DataLoader
    - loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=lambda b: encode_sft_batch(b, tokenizer, device))
    - scheduler = torch.optim.lr_scheduler.LinearLR(
                      optimizer, start_factor=0.1, total_iters=cfg.warmup_iters)
    - Loop until cfg.steps reached, then return.
    """
    # TODO: implement.
    raise NotImplementedError
