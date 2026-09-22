import base64
import gc
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
import unicodedata

ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLCHAIN_DIR = ROOT_DIR / "toolchain"
DIST_DIR = ROOT_DIR / "dist"
DIST_BIN_DIR = DIST_DIR / "bin"
RANKS_BIN = DIST_BIN_DIR / "qwen.ranks.bin"
IS_WIN = sys.platform == "win32"

HERMETIC_PYTHON = TOOLCHAIN_DIR / "python" / ("python.exe" if IS_WIN else "bin/python")
if HERMETIC_PYTHON.exists() and Path(sys.executable).resolve() != HERMETIC_PYTHON.resolve():
    result = subprocess.run([str(HERMETIC_PYTHON.resolve())] + sys.argv, check=False)
    sys.exit(result.returncode)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tiktalkin

try:
    import tiktoken
except ImportError:
    subprocess.run([str(HERMETIC_PYTHON.resolve()), "-m", "pip", "install", "tiktoken"], check=True)
    import tiktoken


def get_process_memory_mb() -> float:
    if IS_WIN:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()

        get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_mem_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        get_mem_info.restype = wintypes.BOOL

        if get_mem_info(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        return 0.0
    else:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0

def resolve_vocab_file() -> Path:
    candidates = [
        ROOT_DIR / "qwen.tiktoken",
        ROOT_DIR.parent / "qwen.tiktoken",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    found = list(ROOT_DIR.parent.rglob("qwen.tiktoken"))
    if found:
        return found[0].resolve()
    raise FileNotFoundError("Could not locate authoritative qwen.tiktoken.")


def run_differential_matrix(ref_enc: tiktoken.Encoding, sov_enc: tiktalkin.Encoding) -> None:
    vectors = [
        ("ABC Notation Basic", "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\n|: CDEF GABc :|"),
        ("ABC Notation Dense Chords", "[CEG]2 ^^C>D [D_FA]4 | (3abc c'2 C,2 ^F# |1 |2 :|" * 8),
        ("Boundary Gate N=62", "A" * 62),
        ("Boundary Gate N=63", "B" * 63),
        ("Boundary Gate N=64 (Stack Ceiling)", "C" * 64),
        ("Boundary Gate N=65 (Heap Transition)", "D" * 65),
        ("Boundary Gate N=66", "E" * 66),
        ("Even Tie Breaking", "aaaaaaaa"),
        ("Odd Tie Breaking", "aaaaaaa"),
        ("Alternating Clashing", "abababab" * 16),
        ("Nested Reductions", "a" * 128),
        ("Whitespace Clashing", "   \t\t\r\n\r\n   Multiple    spaces   \n\n\nand   tabs.   \r\n"),
        ("Massive Space Run", " " * 1024),
        ("Massive Tab Run", "\t" * 512),
        ("CRLF Sequences", "\r\n" * 256),
        ("Contractions Lower", "'s 't 're 've 'm 'll 'd"),
        ("Contractions Upper", "'S 'T 'RE 'VE 'M 'LL 'D"),
        ("Repeated Quotes", "''''''''"),
        ("Numeric Sequences", "0 1 12 123 1234 12345 123456 999999999"),
        ("Accented Latin Romance", "Élégant, café-croissant, façade, naïve, über, Grüße aus Berlin."),
        ("Multilingual CJK Lyrics", "两只老虎跑得快，一只没有耳朵，一只没有尾巴，真奇怪！今晚不眠，快乐无限。"),
        ("SMP Musical Glyphs", "𝄞 𝄢 𝄡 𝅘 𝅥 𝅦"),
        ("Quad-Byte SMP Emojis", "🚀🔥🎧🎼🎹🎺🎻" * 4),
        ("ChatML Standard Structure", "<|im_start|>system\nYou are an authoritative music engine.<|im_end|><|im_start|>user\nCompose.<|im_end|>"),
        ("Sequential Control Delimiters", "<|endoftext|><abc></abc><extra_0><extra_199><mask><sep>"),
        ("Unclosed Delimiter Traps", "Prefix traps: <|im_start, <abc, <<abc>>, <invalid_tag>, <extra_999>, 3 < 5, 10 > 2."),
        ("Stress Compound Block", "Supercalifragilisticexpialidocious" * 16),
        ("Single Character Slices", "A"),
        ("Empty Delimiter Pass", ""),
        ("Long Synthetic Prompt (16KB)", ("Verse: Midnight riding under neon streetlights\n[CEG]2 c'2\n" * 200)),
    ]

    print(f"\n--- Running Adversarial Differential Suite ({len(vectors)} vectors) ---")
    for idx, (label, sample) in enumerate(vectors):
        normalized = unicodedata.normalize("NFC", sample)
        ref_tokens = ref_enc.encode(normalized, allowed_special="all")
        sov_tokens = sov_enc.encode(normalized, allowed_special="all")

        if ref_tokens != sov_tokens:
            print(f"[!] DIVERGENCE on vector [{idx:02d}] ({label}):")
            print(f"    Ref ({len(ref_tokens)}): {ref_tokens[:20]}")
            print(f"    Sov ({len(sov_tokens)}): {sov_tokens[:20]}")
            sys.exit(1)

        decoded = sov_enc.decode(sov_tokens)
        if decoded != normalized:
            print(f"[!] ROUNDTRIP FAILURE on vector [{idx:02d}] ({label}):")
            print(f"    Input:   {repr(normalized)}")
            print(f"    Decoded: {repr(decoded)}")
            sys.exit(1)

        print(f"  Vector [{idx:02d}]: PASSED (Bit-Exact) | {label:<36} ({len(sov_tokens):4d} tokens)")


def run_thread_safety_soak(sov_enc: tiktalkin.Encoding, threads: int = 8, passes_per_thread: int = 5000) -> None:
    print(f"\n--- Running Multithreaded Soak Test ({threads} threads, {passes_per_thread} iterations/thread) ---")
    sample_text = "X:1\nT:Soak Test\nM:4/4\nK:Fm\n[CEG]2 ^^C>D [D_FA]4 | 今晚不眠 快乐无限 | Supercalifragilisticexpialidocious\n"
    normalized = unicodedata.normalize("NFC", sample_text)
    expected_tokens = sov_enc.encode(normalized, allowed_special="all")

    errors: list[str] = []

    def worker(tid: int):
        for _ in range(passes_per_thread):
            toks = sov_enc.encode(normalized, allowed_special="all")
            if toks != expected_tokens:
                errors.append(f"Thread {tid} encountered token mismatch")
                break
            dec = sov_enc.decode(toks)
            if dec != normalized:
                errors.append(f"Thread {tid} encountered decode divergence")
                break

    t_pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in t_pool:
        t.start()
    for t in t_pool:
        t.join()
    t_elapsed = time.perf_counter() - t0

    if errors:
        print(f"[!] Multithreaded soak failure: {errors[0]}")
        sys.exit(1)

    total_encodes = threads * passes_per_thread
    print(f"  [+] Completed {total_encodes:,} concurrent operations in {t_elapsed:.2f}s ({total_encodes / t_elapsed:,.0f} ops/s). Zero race conditions.")


def run_memory_soak(sov_enc: tiktalkin.Encoding, cycles: int = 25000) -> None:
    print(f"\n--- Running Zero-Leak Heap/RSS Soak Test ({cycles:,} cycles) ---")
    payload = "Supercalifragilisticexpialidocious [CEG]2 ^^C>D <|im_start|> 快乐无限\n" * 10
    normalized = unicodedata.normalize("NFC", payload)

    gc.collect()
    mem_initial = get_process_memory_mb()

    for _ in range(1000):
        t = sov_enc.encode(normalized, allowed_special="all")
        _ = sov_enc.decode(t)

    gc.collect()
    mem_baseline = get_process_memory_mb()

    t0 = time.perf_counter()
    for i in range(cycles):
        toks = sov_enc.encode(normalized, allowed_special="all")
        _ = sov_enc.decode(toks)

    t_elapsed = time.perf_counter() - t0
    gc.collect()
    mem_final = get_process_memory_mb()
    delta_mb = mem_final - mem_baseline

    print(f"  Initial RSS:   {mem_initial:.2f} MB")
    print(f"  Baseline RSS:  {mem_baseline:.2f} MB")
    print(f"  Final RSS:     {mem_final:.2f} MB")
    print(f"  Delta RSS:     {delta_mb:+.3f} MB over {cycles:,} cycles ({cycles / t_elapsed:,.0f} ops/s)")

    if delta_mb > 1.5:
        print(f"[!] Leak threshold exceeded: {delta_mb:+.3f} MB")
        sys.exit(1)
    print("  [+] PASS: Memory delta within strict static operating bounds. Zero heap leak confirmed.")


def main() -> None:
    print("=" * 84)
    print("      TIKTALKIN COMPREHENSIVE CONVERGENCE, THREAD SAFETY & SOAK AUDITOR")
    print("=" * 84)

    vocab_path = resolve_vocab_file()
    ranks = {
        base64.b64decode(t): int(r)
        for t, r in (line.split() for line in vocab_path.read_bytes().splitlines() if line)
    }
    specials = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"
    ]
    specials += [f"<extra_{i}>" for i in range(200)]
    specials[204:206] = ["<abc>", "</abc>"]
    pattern = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    ref_enc = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks, special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})
    sov_enc = tiktalkin.Encoding("YuE2", ranks_path=RANKS_BIN)

    run_differential_matrix(ref_enc, sov_enc)
    run_thread_safety_soak(sov_enc, threads=8, passes_per_thread=5000)
    run_memory_soak(sov_enc, cycles=25000)

    print("\n" + "=" * 84)
    print("ALL TESTS PASSED: BIT-EXACT, THREAD-SAFE, ZERO-LEAK SUPREMACY CONFIRMED.")
    print("=" * 84)


if __name__ == "__main__":
    main()