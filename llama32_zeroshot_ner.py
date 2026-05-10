"""
Zero-shot token-level NER with Llama 3.2 Instruct (no training).

For each sentence in an IOB2-like dev file, prompts the model to emit one IOB2
label per token (space-separated). Parses the reply and writes a new file with
the same layout as the input; column index 2 (gold/pred BIO) is replaced.

This script creates ``--out`` only when you run it (nothing writes predictions until then).
Each sentence is written and flushed as it completes.

If ``--out`` already exists, the next run **resumes** after the last fully written sentence
(append mode). Use ``--fresh`` to delete the file and start over.

Long cross-domain jobs (~4k sentences, ~many hours on MPS): run under ``nohup`` so Terminal
can close without killing Python, e.g.
  nohup python3 -u llama32_zeroshot_ner.py --mps --dev splits/cross_domain/dev.iob2 \\
      --out predictions/llama_cd.iob2 >> logs_llama_cd.txt 2>&1 &

Evaluate:
  python span_f1.py en_ewt-ud-dev.iob2 predictions_llama32_zeroshot.iob2

Examples:
  python llama32_zeroshot_ner.py --mps --dev en_ewt-ud-dev.iob2 --max-sentences 20
  python llama32_zeroshot_ner.py --mps --dev en_ewt-ud-dev.iob2
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin" and os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

LABEL_RE = re.compile(r"\b(O|B-PER|I-PER|B-LOC|I-LOC|B-ORG|I-ORG)\b")


def parse_iob2_sentences(path: str):
    """
    Split file into sentences: each item is (comment_lines, token_rows).
    token_rows: list of tab-split field lists (may have >3 columns).
    """
    blocks = []
    comments: list[str] = []
    tokens: list[list[str]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if raw.startswith("#"):
                comments.append(raw)
                continue
            if raw.strip() == "":
                if tokens:
                    blocks.append((comments, tokens))
                    comments, tokens = [], []
                continue
            parts = raw.split("\t")
            if len(parts) >= 3:
                tokens.append(parts)

        if tokens:
            blocks.append((comments, tokens))

    return blocks


def extract_ordered_labels(text: str, n_tokens: int) -> list[str]:
    """Pull IOB2 tags from model output in order; pad/truncate to n_tokens."""
    found = LABEL_RE.findall(text)
    if len(found) >= n_tokens:
        out = list(found[:n_tokens])
    else:
        out = list(found) + ["O"] * (n_tokens - len(found))
    return out


def sentence_from_comments(comments: list[str], fallback: str) -> str:
    for c in comments:
        if c.startswith("# text ="):
            return c.split("=", 1)[1].strip()
    return fallback


def build_user_prompt(sentence_text: str, tokens_space: str) -> str:
    return (
        "Label each token with O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG.\n\n"
        f"Sentence: {sentence_text}\n"
        f"Tokens:   {tokens_space}\n\n"
        "Output one label per token separated by spaces."
    )


def resolve_device(force_cpu: bool, use_mps: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_causal_lm(model_id: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.bfloat16

    kwargs = {}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    model.eval()
    model.to(device)
    return model, tok


def generate_labels(
    model,
    tokenizer,
    device: torch.device,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You annotate named entities in English sentences with IOB2 tags. "
                "Reply with nothing except the requested space-separated labels."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    try:
        batch = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
    except (TypeError, ValueError, KeyError):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out[0, input_ids.shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def write_sentence_block(f, comments, rows, labels: list[str]) -> None:
    for c in comments:
        f.write(c + "\n")
    for row, lab in zip(rows, labels):
        row = list(row)
        while len(row) < 3:
            row.append("")
        row[2] = lab
        f.write("\t".join(row) + "\n")
    f.write("\n")
    f.flush()


def main():
    ap = argparse.ArgumentParser(description="Llama 3.2 zero-shot IOB2 tagging (no training)")
    ap.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-1B-Instruct-bf16",
        help="HF causal LM id (Instruct chat template)",
    )
    ap.add_argument("--dev", default="en_ewt-ud-dev.iob2", help="Input IOB2 dev file")
    ap.add_argument(
        "--out",
        default="predictions_llama32_zeroshot.iob2",
        help="Output IOB2 file (same layout as dev; BIO column replaced)",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument(
        "--mps",
        action="store_true",
        help="Use Apple Metal GPU when available (default without --cpu: CUDA > MPS > CPU)",
    )
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Process only the first N sentences (debug/smoke)",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation budget per sentence (also capped from token count for speed)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore partial output: delete --out and start from sentence 1",
    )
    args = ap.parse_args()

    device = resolve_device(args.cpu, args.mps)
    print(f"Device: {device}  |  Model: {args.model}", flush=True)

    blocks = parse_iob2_sentences(args.dev)
    total = len(blocks)
    if args.max_sentences is not None:
        blocks = blocks[: args.max_sentences]
        total = len(blocks)

    start_i = 0
    if args.fresh and os.path.isfile(args.out):
        os.remove(args.out)
        print(f"--fresh: removed existing {args.out}", flush=True)

    elif os.path.isfile(args.out) and os.path.getsize(args.out) > 0:
        done = parse_iob2_sentences(args.out)
        start_i = len(done)
        if start_i > total:
            raise SystemExit(
                f"{args.out} contains {start_i} sentences but dev has only {total}. "
                "Remove the file or pass --fresh."
            )
        if start_i == total:
            print(
                f"Output already complete ({total} sentences). Nothing to run.\n"
                f"  python span_f1.py {args.dev} {args.out}",
                flush=True,
            )
            return
        if start_i > 0:
            print(
                f"Resuming: found {start_i} complete sentence(s) in {args.out}; "
                f"continuing at {start_i + 1}/{total}",
                flush=True,
            )

    print(f"Sentences to process this run: {total - start_i} (of {total} total)  →  {args.out}", flush=True)

    model, tokenizer = load_causal_lm(args.model, device)

    file_mode = "a" if start_i > 0 else "w"
    written_this_run = 0
    with open(args.out, file_mode, encoding="utf-8") as fout:
        for i in range(start_i, total):
            comments, rows = blocks[i]
            toks = [r[1] for r in rows]
            tokens_line = " ".join(toks)
            sentence_text = sentence_from_comments(comments, tokens_line)
            prompt = build_user_prompt(sentence_text, tokens_line)

            # Shorter generations for short sentences → fewer multi-hour runs
            dyn_cap = min(args.max_new_tokens, max(48, len(toks) * 6 + 32))

            raw = generate_labels(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=dyn_cap,
            )
            labels = extract_ordered_labels(raw, len(toks))

            if len(labels) != len(toks):
                labels = (labels + ["O"] * len(toks))[: len(toks)]

            write_sentence_block(fout, comments, rows, labels)
            written_this_run += 1

            if device.type == "mps" and (i + 1) % 150 == 0:
                torch.mps.empty_cache()

            if (i + 1) % 50 == 0 or i == start_i:
                print(
                    f"  [{i + 1}/{total}] tokens={len(toks)}  (streaming → {args.out})",
                    flush=True,
                )

    print(f"\nWrote {written_this_run} sentence(s) this run ({total} total in {args.out})", flush=True)
    print("Evaluate:", flush=True)
    print(f"  python span_f1.py {args.dev} {args.out}", flush=True)


if __name__ == "__main__":
    main()
