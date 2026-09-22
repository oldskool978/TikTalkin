import base64
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLCHAIN_DIR = ROOT_DIR / "toolchain"
IS_WIN = sys.platform == "win32"

HERMETIC_PYTHON = TOOLCHAIN_DIR / "python" / (
    "python.exe" if IS_WIN else "bin/python"
)
if (
    HERMETIC_PYTHON.exists()
    and Path(sys.executable).resolve() != HERMETIC_PYTHON.resolve()
):
  result = subprocess.run(
      [str(HERMETIC_PYTHON.resolve())] + sys.argv, check=False
  )
  sys.exit(result.returncode)

if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

try:
  import tiktoken
except ImportError:
  subprocess.run(
      [
          str(HERMETIC_PYTHON.resolve()),
          "-m",
          "pip",
          "install",
          "--no-warn-script-location",
          "tiktoken",
      ],
      check=True,
  )
  import tiktoken

import tiktalkin

CORPUS = [
    (
        "Doodle ABC Prompt (64KB)",
        (
            "X:1\nT:Tonight Awake\nM:4/4\nL:1/8\nQ:1/4=120\nK:Fm\n|: [CEG]2"
            " ^^C>D [D_FA]4 | (3abc c'2 C,2 ^F# |1 |2 :|\n"
            * 600
        ),
    ),
    (
        "Vocal Lyrics UTF-8 Multilingual (64KB)",
        (
            "Midnight riding under neon streetlights\n"
            "Searching for the answers in the rearview mirror\n"
            "Élégant, café-croissant, façade, naïve, façade, über, Grüße aus"
            " Berlin.\n"
            "两只老虎跑得快，一只没有耳朵，一只没有尾巴，真奇怪！\n"
            * 250
        ),
    ),
    (
        "Stress Token Block (64KB)",
        "Supercalifragilisticexpialidocious " * 1900,
    ),
]


def resolve_vocab_file() -> Path:
  vocab_source = ROOT_DIR / "qwen.tiktoken"
  if not vocab_source.exists():
    found = list(ROOT_DIR.parent.rglob("qwen.tiktoken"))
    if found:
      return found[0]
  return vocab_source


def run_benchmark():
  print("=" * 84)
  print(
      "       SOVEREIGN TOKENIZER BENCHMARK: TIKTALKIN (C11 SIMD) vs TIKTOKEN"
  )
  print("=" * 84)

  vocab_path = resolve_vocab_file()
  if not vocab_path.exists():
    raise FileNotFoundError(f"Missing authoritative vocabulary: {vocab_path}")

  ranks = {
      base64.b64decode(t): int(r)
      for t, r in (
          line.split() for line in vocab_path.read_bytes().splitlines() if line
      )
  }
  specials = [
      "<|endoftext|>",
      "<|im_start|>",
      "<|im_end|>",
      "<R>",
      "<S>",
      "<X>",
      "<mask|",
      "<sep>",
  ]
  specials += [f"<extra_{i}>" for i in range(200)]
  specials[204:206] = ["<abc>", "</abc>"]
  pattern = (
      r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
      r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  )

  ref = tiktoken.Encoding(
      "YuE2",
      pat_str=pattern,
      mergeable_ranks=ranks,
      special_tokens={s: i + len(ranks) for i, s in enumerate(specials)},
  )
  sov = tiktalkin.Encoding("YuE2")

  for name, text in CORPUS:
    normalized = unicodedata.normalize("NFC", text)
    raw_bytes = len(normalized.encode("utf-8"))

    ref_out = ref.encode_ordinary(normalized)
    sov_out = sov.encode_ordinary(normalized)
    assert (
        ref_out == sov_out
    ), f"Parity mismatch on {name}: Ref={len(ref_out)}, Sov={len(sov_out)}"

    for _ in range(5):
      ref.encode_ordinary(normalized)
      sov.encode_ordinary(normalized)

    iters = 40
    t0 = time.perf_counter()
    for _ in range(iters):
      ref.encode_ordinary(normalized)
    t_ref = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
      sov.encode_ordinary(normalized)
    t_sov = (time.perf_counter() - t0) / iters

    mb = raw_bytes / (1024 * 1024)
    speedup = t_ref / t_sov
    mb_s_sov = mb / t_sov
    mb_s_ref = mb / t_ref

    print(
        f"\n--- Target: {name} ({raw_bytes:,} bytes, {len(sov_out):,} tokens)"
        " ---"
    )
    print(
        f"  Reference tiktoken: {t_ref * 1000.0:7.3f} ms | {mb_s_ref:8.2f} MB/s"
        f" | {len(ref_out) / t_ref:11,.0f} tokens/s"
    )
    print(
        f"  Sovereign TikTalkin:{t_sov * 1000.0:7.3f} ms | {mb_s_sov:8.2f} MB/s"
        f" | {len(sov_out) / t_sov:11,.0f} tokens/s"
    )
    print(
        f"  Acceleration:       {speedup:.2f}x faster"
        f" ({'+' if speedup >= 1.0 else ''}{(speedup - 1.0)*100.0:.1f}%)"
    )

  print("\n" + "=" * 84)
  print("ALL DIFFERENTIAL CHECKS CONFIRMED: 100% BIT-PARITY ACHIEVED")
  print("=" * 84)


if __name__ == "__main__":
  run_benchmark()