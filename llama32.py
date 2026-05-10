"""
Fine-tune Llama 3.2 (1B) for token-level NER on EWT IOB2 — same protocol as baseline.py.

Default checkpoint is an open HF mirror of Llama 3.2 1B Instruct (no Meta portal login).
Override with `--model meta-llama/Llama-3.2-1B` if you have accepted the license and set HF_TOKEN.

On Apple Silicon (MPS), training uses micro-batch **1** plus gradient accumulation — larger
micro-batches routinely OOM for 1B models.

**Recommended on Mac (real NER quality, fits unified memory):** LoRA adapters (`pip install peft`):
  python llama32.py --mps --lora --max-seq-length 96 --train-batch-size 2 \\
      --eval-batch-size 2 --epochs 2 --early-stopping-patience 2

Standard EWT fine-tune + dev predictions (3 epochs):
  python llama32.py --mps --lora --train en_ewt-ud-train.iob2 \\
      --dev en_ewt-ud-dev.iob2 --out predictions/llama_finetuned_standard.iob2 \\
      --epochs 3 --train-batch-size 2 --eval-batch-size 2 --max-seq-length 96

Shorthand for the same paths:
  python llama32.py --mps --lora --split standard --epochs 3 \\
      --train-batch-size 2 --eval-batch-size 2 --max-seq-length 96

All experimental splits under ``splits/*/`` (four sequential jobs, separate checkpoints):
  python llama32.py --mps --lora --run-all-splits --epochs 3 \\
      --train-batch-size 2 --eval-batch-size 2 --max-seq-length 96

Quick pipeline smoke (weak F1 — linear head only):
  python llama32.py --mps --head-only --max-train-samples 300 --epochs 1 \\
      --train-batch-size 2 --max-seq-length 96 --early-stopping-patience 0

Llama **3.1 8B** (`llama31_8b.py`) is *not* faster on Apple Silicon — ~8× larger, slower,
and more likely to OOM. Use **this 3.2 1B** script on Mac; reserve 8B for CUDA servers.

Usage:
  python llama32.py --mps
  python llama32.py --cpu --epochs 2 --train-batch-size 4
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if sys.platform == "darwin" and os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    LoraConfig = get_peft_model = TaskType = None

LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

DEFAULT_TRAIN = "en_ewt-ud-train.iob2"
DEFAULT_DEV = "en_ewt-ud-dev.iob2"
DEFAULT_OUT = "predictions_llama32.iob2"
DEFAULT_TEST = "en_ewt-ud-test-masked.iob2"

EXPERIMENTAL_SPLIT_NAMES = (
    "entity_disjoint",
    "frequency_adv",
    "context_shift",
    "cross_domain",
)

SPLIT_PRESETS: dict[str, tuple[str, str, str]] = {
    "standard": (
        DEFAULT_TRAIN,
        DEFAULT_DEV,
        "predictions/llama_finetuned_standard.iob2",
    ),
}
for _name in EXPERIMENTAL_SPLIT_NAMES:
    SPLIT_PRESETS[_name] = (
        f"splits/{_name}/train.iob2",
        f"splits/{_name}/dev.iob2",
        f"predictions/llama_finetuned_{_name}.iob2",
)


def _sanitize_run_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Llama 3.2 token classification NER on EWT")
    ap.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-1B-Instruct-bf16",
        help="HF model id (default: open Llama-3.2-1B Instruct mirror)",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU training")
    ap.add_argument(
        "--mps",
        action="store_true",
        help="Try Apple Metal GPU; default on Mac without this is CPU",
    )
    ap.add_argument(
        "--epochs", type=int, default=2, help="Training epochs (early stopping may end sooner)"
    )
    ap.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device train batch (lower if OOM); on MPS this is simulated via grad accumulation",
    )
    ap.add_argument("--eval-batch-size", type=int, default=8, help="Per-device eval batch")
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop if eval F1 does not improve for this many epochs (0 = disabled)",
    )
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training sentences (for smoke tests; omit for full runs)",
    )
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        metavar="N",
        help="Tokenizer max length (96 or 64 uses less MPS memory than 128)",
    )
    ap.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze Llama weights; train only the token-classification head (much less RAM — "
        "good for MPS smoke tests; not full fine-tuning)",
    )
    ap.add_argument(
        "--lora",
        action="store_true",
        help="Train LoRA adapters (+ classifier head). Strongly recommended on MPS instead of "
        "full fine-tuning (needs: pip install peft).",
    )
    ap.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default 8; lower if OOM)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-4 with --lora, else 2e-5)",
    )
    ap.add_argument(
        "--train",
        default=None,
        metavar="PATH",
        help=f"IOB2 train file (default: {DEFAULT_TRAIN}; overridden by --split)",
    )
    ap.add_argument(
        "--dev",
        default=None,
        metavar="PATH",
        help=f"IOB2 dev file (default: {DEFAULT_DEV}; overridden by --split)",
    )
    ap.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help=f"Dev predictions IOB2 (default: {DEFAULT_OUT}; overridden by --split)",
    )
    ap.add_argument(
        "--test",
        default=None,
        metavar="PATH",
        help=f"IOB2 test file for masked-label inference (default: {DEFAULT_TEST}; use --skip-test to disable)",
    )
    ap.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not load or predict on the test set (useful if the default test file is missing)",
    )
    ap.add_argument(
        "--split",
        choices=sorted(SPLIT_PRESETS.keys()),
        default=None,
        help="Fill default --train / --dev / --out for standard EWT or an experimental split "
        "(explicit paths still win)",
    )
    ap.add_argument(
        "--run-all-splits",
        action="store_true",
        help=f"Run training sequentially for: {', '.join(EXPERIMENTAL_SPLIT_NAMES)} "
        "(plus standard EWT if --include-standard)",
    )
    ap.add_argument(
        "--include-standard",
        action="store_true",
        help="With --run-all-splits, also train/predict on standard EWT first",
    )
    args = ap.parse_args(argv)

    if args.split:
        pt, pd, po = SPLIT_PRESETS[args.split]
        if args.train is None:
            args.train = pt
        if args.dev is None:
            args.dev = pd
        if args.out is None:
            args.out = po
    if args.train is None:
        args.train = DEFAULT_TRAIN
    if args.dev is None:
        args.dev = DEFAULT_DEV
    if args.out is None:
        args.out = DEFAULT_OUT

    if args.skip_test:
        args.test = None
    elif args.test is None:
        args.test = DEFAULT_TEST

    if args.head_only and args.lora:
        raise SystemExit("Choose either --head-only or --lora, not both.")
    if args.lora and get_peft_model is None:
        raise SystemExit("LoRA requires the `peft` package. Install with:\n  pip install peft")
    if args.run_all_splits and args.split:
        raise SystemExit("Use either --split (single job) or --run-all-splits, not both.")

    return args


def load_ner_model_torch_dtype(model_checkpoint: str, torch_dtype):
    """Load token-classification head; SDPA when the installed stack supports it."""
    common = dict(
        pretrained_model_name_or_path=model_checkpoint,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch_dtype,
    )
    try:
        return AutoModelForTokenClassification.from_pretrained(
            **common,
            attn_implementation="sdpa",
        )
    except (TypeError, ValueError, RuntimeError):
        return AutoModelForTokenClassification.from_pretrained(**common)


def read_ewt(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"IOB2 file not found: {path}")
    sentences = []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.strip() == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split("\t")
            tag_id = LABEL2ID.get(parts[2], LABEL2ID["O"])
            tokens.append(parts[1])
            tags.append(tag_id)
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


def write_iob2_with_predictions(
    source_path,
    sentences_data,
    prediction_logits,
    output_path,
    tokenizer,
    label_list,
    max_seq_length,
):
    pred_ids = np.argmax(prediction_logits, axis=2)
    n_sent = len(sentences_data)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(source_path, encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        sent_idx = 0
        for line in f_in:
            line = line.rstrip("\n")
            if line.startswith("#"):
                f_out.write(line + "\n")
                continue
            if line.strip() == "":
                f_out.write("\n")
                sent_idx += 1
                continue
            parts = line.split("\t")
            if sent_idx >= n_sent:
                raise ValueError(
                    f"More token lines in {source_path} than sentences ({n_sent})"
                )
            sentence_preds = pred_ids[sent_idx]
            tokens_in_sent = sentences_data[sent_idx]["tokens"]
            encoding = tokenizer(
                tokens_in_sent,
                truncation=True,
                max_length=max_seq_length,
                is_split_into_words=True,
            )
            word_ids = encoding.word_ids()
            seen, word_pred_map = set(), {}
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    word_pred_map[wid] = label_list[sentence_preds[pos]]
            word_pos = int(parts[0]) - 1
            parts[2] = word_pred_map.get(word_pos, "O")
            f_out.write("\t".join(parts) + "\n")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions, true_labels = [], []
    for pred, lab in zip(predictions, labels):
        curr_preds, curr_labels = [], []
        for p, l in zip(pred, lab):
            if l != -100:
                curr_preds.append(LABEL_LIST[p])
                curr_labels.append(LABEL_LIST[l])
        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def run_single_job(args: argparse.Namespace, *, job_name: str) -> None:
    """Train once for the given train/dev/out paths (and optional test)."""
    model_checkpoint = args.model
    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    output_file = args.out
    learning_rate = args.lr if args.lr is not None else (1e-4 if args.lora else 2e-5)

    run_id = _sanitize_run_id(job_name)
    output_dir = f"./results_llama32_{run_id}"
    test_output_stem = os.path.splitext(os.path.basename(output_file))[0]
    test_output_file = f"test_predictions_{test_output_stem}.iob2"

    max_seq_length = args.max_seq_length
    num_train_epochs = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    print("SCRIPT STARTED — Llama 3.2 token classification NER")
    print(f"Job   : {job_name}")
    print(f"Model : {model_checkpoint}")
    print(f"Train : {train_file}  |  Dev : {dev_file}  |  Out : {output_file}")
    print(f"Ckpt dir: {output_dir}")
    print(
        f"Epochs: {num_train_epochs}  |  Batch: {train_batch_size}/{eval_batch_size}  "
        f"|  max_seq_length: {max_seq_length}  |  lr: {learning_rate}  "
        f"|  lora: {args.lora}  |  head_only: {args.head_only}"
    )
    set_seed(42)

    print("Loading IOB2 data...")
    train_data = read_ewt(train_file)
    dev_data = read_ewt(dev_file)
    test_data = read_ewt(test_file) if test_file else []

    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data) if test_data else None

    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(n))
        print(f"  (subset) training on first {n} sentences (--max-train-samples)")

    print(f"  train sentences : {len(train_dataset)}")
    print(f"  dev   sentences : {len(dev_dataset)}")
    if test_dataset is not None:
        print(f"  test  sentences : {len(test_dataset)}")
    else:
        print("  test  : skipped (--skip-test or no --test)")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=max_seq_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            prev_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing...")
    remove_cols = ["tokens", "ner_tags"]
    tokenized_train = train_dataset.map(
        tokenize_and_align_labels, batched=True, remove_columns=remove_cols
    )
    tokenized_dev = dev_dataset.map(
        tokenize_and_align_labels, batched=True, remove_columns=remove_cols
    )
    tokenized_test = (
        test_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=remove_cols)
        if test_dataset is not None
        else None
    )
    print("Tokenization done")

    cuda_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()

    def device_training_plan():
        if args.cpu:
            return (
                True,
                train_batch_size,
                eval_batch_size,
                1,
                True,
                "CPU (forced)",
                torch.float32,
                False,
                False,
            )
        if cuda_avail:
            bf16_ok = torch.cuda.is_bf16_supported()
            load_dt = torch.bfloat16 if bf16_ok else torch.float16
            return (
                False,
                train_batch_size,
                eval_batch_size,
                1,
                True,
                "CUDA",
                load_dt,
                bf16_ok,
                not bf16_ok,
            )
        if mps_avail:
            if not args.mps:
                return (
                    True,
                    train_batch_size,
                    eval_batch_size,
                    1,
                    True,
                    "CPU (Apple Silicon default — pass --mps to try GPU)",
                    torch.float32,
                    False,
                    False,
                )
            micro = 1
            accum = max(1, train_batch_size)
            eval_bs = max(1, min(eval_batch_size, 4))
            return (
                False,
                micro,
                eval_bs,
                accum,
                True,
                "MPS (micro=1 + grad accum + checkpoint)",
                torch.bfloat16,
                False,
                False,
            )
        return (
            True,
            train_batch_size,
            eval_batch_size,
            1,
            True,
            "CPU",
            torch.float32,
            False,
            False,
        )

    (
        use_cpu,
        train_bs,
        eval_bs,
        grad_accum,
        grad_ckpt,
        plan_desc,
        load_dtype,
        train_bf16,
        train_fp16,
    ) = device_training_plan()

    print("Loading model...")
    model = load_ner_model_torch_dtype(model_checkpoint, load_dtype)

    def maybe_freeze_backbone(m):
        if not args.head_only:
            return
        n_trainable = 0
        for name, param in m.named_parameters():
            if name.startswith("score."):
                param.requires_grad = True
                n_trainable += param.numel()
            else:
                param.requires_grad = False
        print(
            f"  --head-only: training classifier head only (~{n_trainable:,} trainable params)",
            flush=True,
        )

    maybe_freeze_backbone(model)

    def wrap_peft_if_requested(m):
        if not args.lora:
            return m
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=max(args.lora_r * 2, 16),
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        out = get_peft_model(m, peft_config)
        out.print_trainable_parameters()
        return out

    model = wrap_peft_if_requested(model)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    def make_trainer(m):
        if grad_ckpt:
            m.gradient_checkpointing_enable()
        elif hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable()
        early_cb = []
        save_kw = dict(save_strategy="no")
        best_kw = dict(load_best_model_at_end=False, metric_for_best_model=None)
        if args.early_stopping_patience > 0:
            early_cb = [
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
            ]
            save_kw = dict(save_strategy="epoch", save_total_limit=1)
            best_kw = dict(
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
            )
        targs = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=grad_ckpt,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            report_to="none",
            fp16=train_fp16 and not use_cpu,
            bf16=train_bf16 and not use_cpu,
            use_cpu=use_cpu,
            dataloader_pin_memory=cuda_avail,
            dataloader_num_workers=0,
            **save_kw,
            **best_kw,
        )
        return Trainer(
            model=m,
            args=targs,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=early_cb,
        )

    trainer = make_trainer(model)
    print(
        f"Training plan: {plan_desc} | train_bs={train_bs} "
        f"accum={grad_accum} eval_bs={eval_bs} grad_ckpt={grad_ckpt} "
        f"amp_bf16={train_bf16} amp_fp16={train_fp16} dtype={load_dtype}",
        flush=True,
    )

    print("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        msg = str(e).lower()
        oom_mps = (
            ("out of memory" in msg or "mps backend" in msg)
            and mps_avail
            and not use_cpu
            and not cuda_avail
        )
        if not oom_mps:
            raise
        cpu_steps = math.ceil(len(tokenized_train) / max(1, train_batch_size)) * num_train_epochs
        if cpu_steps > 80:
            raise RuntimeError(
                "MPS ran out of memory and CPU fallback was skipped because it would take "
                "many hours (~minutes per step for 1B).\n"
                "Retry with less memory pressure, for example:\n"
                "  --lora   # strongly recommended on Mac (pip install peft)\n"
                "  --max-seq-length 96   (or 64)\n"
                "  --lora-r 4\n"
                "  --max-train-samples 500 --epochs 1 --train-batch-size 2\n"
                "Or use --head-only for a fast wiring check only.\n"
                "Full 8B (`llama31_8b.py`) needs CUDA — not faster on Apple Silicon."
            ) from e
        print(
            f"MPS OOM — reloading model on CPU (~{cpu_steps} optimizer steps; expect long runtime).",
            flush=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        del trainer
        model = load_ner_model_torch_dtype(model_checkpoint, torch.float32)
        maybe_freeze_backbone(model)
        model = wrap_peft_if_requested(model)
        use_cpu_local = True
        train_bs_fb, eval_bs_fb = train_batch_size, eval_batch_size
        grad_accum_fb = 1
        grad_ckpt_fb = True
        train_bf16_fb = train_fp16_fb = False
        load_dtype_fb = torch.float32
        plan_desc_fb = "CPU (fallback after MPS OOM)"

        def make_trainer_fallback(m):
            if grad_ckpt_fb:
                m.gradient_checkpointing_enable()
            elif hasattr(m, "gradient_checkpointing_disable"):
                m.gradient_checkpointing_disable()
            early_cb = []
            save_kw = dict(save_strategy="no")
            best_kw = dict(load_best_model_at_end=False, metric_for_best_model=None)
            if args.early_stopping_patience > 0:
                early_cb = [
                    EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
                ]
                save_kw = dict(save_strategy="epoch", save_total_limit=1)
                best_kw = dict(
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                )
            targs = TrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                per_device_train_batch_size=train_bs_fb,
                per_device_eval_batch_size=eval_bs_fb,
                gradient_accumulation_steps=grad_accum_fb,
                gradient_checkpointing=grad_ckpt_fb,
                num_train_epochs=num_train_epochs,
                weight_decay=0.01,
                eval_strategy="epoch",
                logging_strategy="steps",
                logging_steps=100,
                report_to="none",
                fp16=False,
                bf16=False,
                use_cpu=True,
                dataloader_pin_memory=cuda_avail,
                dataloader_num_workers=0,
                **save_kw,
                **best_kw,
            )
            return Trainer(
                model=m,
                args=targs,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_dev,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=early_cb,
            )

        trainer = make_trainer_fallback(model)
        print(
            f"Training plan: {plan_desc_fb} | train_bs={train_bs_fb} accum={grad_accum_fb} "
            f"eval_bs={eval_bs_fb} grad_ckpt={grad_ckpt_fb} dtype={load_dtype_fb}",
            flush=True,
        )
        trainer.train()

    print("\nEvaluating on dev set...")
    dev_predictions, dev_labels, _ = trainer.predict(tokenized_dev)
    results = compute_metrics((dev_predictions, dev_labels))

    print("\n── LLAMA 3.2 RESULTS (dev) ──")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\nSaving dev predictions to {output_file} ...")
    write_iob2_with_predictions(
        dev_file,
        dev_data,
        dev_predictions,
        output_file,
        tokenizer,
        LABEL_LIST,
        max_seq_length,
    )
    print(f"Predictions saved to {output_file}")
    print(f"\nSpan F1 (strict): python span_f1.py {dev_file} {output_file}")

    if tokenized_test is not None:
        print("\nTest set has masked NER labels; skipping F1 on test.")
        test_predictions, _, _ = trainer.predict(tokenized_test)
        print(f"Saving test predictions to {test_output_file} ...")
        write_iob2_with_predictions(
            test_file,
            test_data,
            test_predictions,
            test_output_file,
            tokenizer,
            LABEL_LIST,
            max_seq_length,
        )
        print(f"Test predictions saved to {test_output_file}")
    else:
        print("\nSkipping test predictions (--skip-test).")

    if mps_avail and args.mps:
        torch.mps.empty_cache()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_all_splits:
        queue: list[tuple[str, str, str, str]] = []
        if args.include_standard:
            t, d, o = SPLIT_PRESETS["standard"]
            queue.append(("standard", t, d, o))
        for name in EXPERIMENTAL_SPLIT_NAMES:
            t, d, o = SPLIT_PRESETS[name]
            queue.append((name, t, d, o))
        for job_name, tr, de, ou in queue:
            sub = argparse.Namespace(**vars(args))
            sub.train = tr
            sub.dev = de
            sub.out = ou
            sub.split = None
            sub.run_all_splits = False
            print("\n" + "=" * 72)
            print(f" QUEUE: {job_name} ")
            print("=" * 72 + "\n")
            run_single_job(sub, job_name=job_name)
        print("\nAll queued split jobs finished.")
        return

    job_name = args.split or os.path.splitext(os.path.basename(args.out))[0]
    run_single_job(args, job_name=job_name)


if __name__ == "__main__":
    main()
