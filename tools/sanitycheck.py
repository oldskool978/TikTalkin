from __future__ import annotations

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
        ("Sequential Control Delimiters", "<|endoftext|><abc></abc><extra_0><extra_199><mask ><sep>"),
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


def run_special_token_edgecases(ref_enc: tiktoken.Encoding, sov_enc: tiktalkin.Encoding) -> None:
    print("\n--- Running Special Token Edge-Case & Subset Verification ---")

    assert "<mask >" in sov_enc.special_tokens_set, "Assertion Error: '<mask >' missing from special_tokens_set"
    assert "<mask|>" not in sov_enc.special_tokens_set, "Assertion Error: Outdated '<mask|>' typo still present"
    assert "<music_start>" in sov_enc.special_tokens_set, "Assertion Error: '<music_start>' missing"
    assert "<music_end>" in sov_enc.special_tokens_set, "Assertion Error: '<music_end>' missing"

    try:
        ref_enc.encode("This should fail: <|endoftext|>", disallowed_special="all")
        raise AssertionError("ref_enc failed to raise ValueError")
    except ValueError:
        pass

    try:
        sov_enc.encode("This should fail: <|endoftext|>", disallowed_special="all")
        raise AssertionError("sov_enc failed to raise ValueError")
    except ValueError:
        print("  [+] Disallowed token rejection: PASSED (ValueError raised identically to tiktoken)")

    test_text = "Instruction: generate score <abc>X:1|C4</abc> with <|endoftext|> delimiter."
    allowed_subsets = [
        {"<abc>", "</abc>"},
        {"<|endoftext|>"},
        {"<abc>", "</abc>", "<|endoftext|>"},
    ]

    for subset in allowed_subsets:
        ref_out = ref_enc.encode(test_text, allowed_special=subset, disallowed_special=())
        sov_out = sov_enc.encode(test_text, allowed_special=subset, disallowed_special=())
        assert ref_out == sov_out, f"Subset mismatch on allowed_special={subset}: {ref_out} != {sov_out}"
        print(f"  [+] Subset verification: PASSED (Bit-Exact match for {subset})")

    adjacent_test = "<abc><|endoftext|></abc>"
    ref_adj = ref_enc.encode(adjacent_test, allowed_special={"<abc>", "</abc>"}, disallowed_special=())
    sov_adj = sov_enc.encode(adjacent_test, allowed_special={"<abc>", "</abc>"}, disallowed_special=())
    assert ref_adj == sov_adj, f"Adjacent special mismatch: {ref_adj} != {sov_adj}"
    assert 151643 not in sov_adj, "<|endoftext|> was wrongly tokenized as special ID 151643"
    print("  [+] Adjacent unallowed special treated as ordinary text: PASSED (Bit-Exact)")


def run_agentic_prefix_splicing_audit(sov_enc: tiktalkin.Encoding) -> None:
    print("\n--- Running Agentic Prefix Splicing Audit ---")
    
    a1 = [10, 20, 30, 40, 50]
    b1 = [10, 20, 30, 40, 50]
    assert tiktalkin.find_common_prefix(a1, b1) == 5, "Identical sequence prefix mismatch"

    a2 = [10, 20, 30, 40, 50]
    b2 = [10, 20, 30, 99, 50]
    assert tiktalkin.find_common_prefix(a2, b2) == 3, "Divergence calculation failure"

    assert tiktalkin.find_common_prefix([], [1, 2, 3]) == 0, "Empty sequence boundary failure"

    orig_abc = "X:1\nM:4/4\nL:1/8\nK:C\n|: CDEF GABc | [CEG]4 c4 :|"
    edit_abc = "X:1\nM:4/4\nL:1/8\nK:C\n|: CDEF GABc | [CFA]4 c4 :|"
    
    t_orig = sov_enc.encode_ordinary(orig_abc)
    t_edit = sov_enc.encode_ordinary(edit_abc)
    split_pt = tiktalkin.find_common_prefix(t_orig, t_edit)
    
    assert 0 < split_pt < len(t_orig), "ABC mutation split point out of range"
    assert t_orig[:split_pt] == t_edit[:split_pt], "Common prefix content mismatch"
    assert t_orig[split_pt] != t_edit[split_pt], "Common prefix divergence failure"
    
    saved_ratio = (split_pt / float(len(t_orig))) * 100.0
    print(f"  [+] Realistic score mutation: PASSED ({split_pt}/{len(t_orig)} tokens retained, {saved_ratio:.1f}% KV-cache reuse)")


def run_batch_pipeline_audit(ref_enc: tiktoken.Encoding, sov_enc: tiktalkin.Encoding) -> None:
    print("\n--- Running Multithreaded Batch Pipeline Audit ---")
    batch_prompts = [
        "X:1\nT:Batch 1\nM:4/4\nK:C\nCDEF GABc |",
        "Verse: Midnight riding under neon streetlights",
        "夜空中最亮的星 能否听清",
        "<|im_start|>system\nCompose a song.<|im_end|>",
        "Supercalifragilisticexpialidocious",
    ]

    ref_batch = [ref_enc.encode(p, allowed_special="all") for p in batch_prompts]
    sov_batch = sov_enc.encode_batch(batch_prompts, allowed_special="all", num_threads=4)
    assert ref_batch == sov_batch, "encode_batch returned divergent output"

    sov_decoded_batch = sov_enc.decode_batch(sov_batch, num_threads=4)
    assert sov_decoded_batch == batch_prompts, "decode_batch roundtrip divergence"
    print("  [+] Concurrent batch encoding and decoding: PASSED (100% bit-exact across threads)")


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
    specials_dict = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<R>": 151646,
        "<S>": 151647,
        "<X>": 151648,
        "<mask >": 151649,
        "<sep>": 151650,
    }
    for i in range(196):
        specials_dict[f"<extra_{i}>"] = 151651 + i
    specials_dict["<abc>"] = 151847
    specials_dict["</abc>"] = 151848
    specials_dict["<extra_198>"] = 151849
    specials_dict["<extra_199>"] = 151850
    specials_dict["<music_start>"] = 151851
    specials_dict["<music_end>"] = 151852
    specials_dict["<latent_start>"] = 184621
    specials_dict["<latent_end>"] = 184622
    specials_dict["<latent_pad>"] = 184623

    pattern = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    ref_enc = tiktoken.Encoding(
        "YuE2",
        pat_str=pattern,
        mergeable_ranks=ranks,
        special_tokens=specials_dict,
    )
    
    with tiktalkin.Encoding("YuE2", ranks_path=RANKS_BIN) as sov_enc:
        run_differential_matrix(ref_enc, sov_enc)
        run_special_token_edgecases(ref_enc, sov_enc)
        run_agentic_prefix_splicing_audit(sov_enc)
        run_batch_pipeline_audit(ref_enc, sov_enc)
        run_thread_safety_soak(sov_enc, threads=8, passes_per_thread=5000)
        run_memory_soak(sov_enc, cycles=25000)

    print("\n" + "=" * 84)
    print("ALL TESTS PASSED: BIT-EXACT, THREAD-SAFE, ZERO-LEAK SUPREMACY CONFIRMED.")
    print("=" * 84)


if __name__ == "__main__":
    main()