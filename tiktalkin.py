"""
TikTalkin Sovereign Tokenizer Python Client
Zero-dependency, standalone drop-in replacement for tiktoken with O(1) SIMD lookup.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Union

ROOT_DIR = Path(__file__).resolve().parent
IS_WIN = sys.platform == "win32"
_DLL_NAME = "tiktalkin.dll" if IS_WIN else "libtiktalkin.so"


class Telemetry(ctypes.Structure):
    _fields_ = [
        ("total_tokens", ctypes.c_uint32),
        ("table_capacity", ctypes.c_uint32),
        ("sso_tokens", ctypes.c_uint32),
        ("max_psl", ctypes.c_uint32),
        ("string_blob_bytes", ctypes.c_uint64),
        ("load_factor", ctypes.c_double),
        ("sso_ratio", ctypes.c_double),
        ("mean_psl", ctypes.c_double),
        ("psl_variance", ctypes.c_double),
        ("compile_seconds", ctypes.c_double),
    ]


class _TikTalkinBinding:
    def __init__(self, dll_path: Path):
        self.lib = ctypes.CDLL(str(dll_path))

        self.lib.tiktalkin_init.argtypes = [ctypes.c_char_p]
        self.lib.tiktalkin_init.restype = ctypes.c_void_p

        self.lib.tiktalkin_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
        ]
        self.lib.tiktalkin_encode.restype = ctypes.c_int32

        self.lib.tiktalkin_encode_ordinary.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
        ]
        self.lib.tiktalkin_encode_ordinary.restype = ctypes.c_int32

        self.lib.tiktalkin_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self.lib.tiktalkin_decode.restype = ctypes.c_int32

        self.lib.tiktalkin_destroy.argtypes = [ctypes.c_void_p]
        self.lib.tiktalkin_destroy.restype = None

        self.lib.tiktalkin_compile_vocab_with_telemetry.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(Telemetry),
        ]
        self.lib.tiktalkin_compile_vocab_with_telemetry.restype = ctypes.c_int32


def _locate_native_library() -> Path:
    candidates = [
        ROOT_DIR / _DLL_NAME,
        ROOT_DIR / "dist" / "bin" / _DLL_NAME,
        ROOT_DIR / "build" / _DLL_NAME,
        Path(os.environ.get("TIKTALKIN_DLL_PATH", "")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Unable to locate {_DLL_NAME}. Ensure the library is compiled in dist/bin/ or adjacent to tiktalkin.py."
    )


_BINDING: Optional[_TikTalkinBinding] = None


def get_binding() -> _TikTalkinBinding:
    global _BINDING
    if _BINDING is None:
        _BINDING = _TikTalkinBinding(_locate_native_library())
    return _BINDING


def compile_vocab(
    in_tiktoken_path: Union[str, Path],
    out_bin_path: Union[str, Path],
    out_telemetry_json: Optional[Union[str, Path]] = None,
) -> Telemetry:
    binding = get_binding()
    in_path_str = str(Path(in_tiktoken_path).resolve()).encode("utf-8")
    out_path_str = str(Path(out_bin_path).resolve()).encode("utf-8")
    telem_path_str = (
        str(Path(out_telemetry_json).resolve()).encode("utf-8")
        if out_telemetry_json
        else None
    )

    telem = Telemetry()
    rc = binding.lib.tiktalkin_compile_vocab_with_telemetry(
        in_path_str,
        out_path_str,
        telem_path_str,
        ctypes.byref(telem),
    )
    if rc != 0:
        raise RuntimeError(f"Vocabulary compilation failed with error code: {rc}")
    return telem


def compile_qwen_ranks_binary(
    source_dir_or_file: Union[str, Path],
    target_bin_path: Optional[Union[str, Path]] = None,
) -> Path:
    src = Path(source_dir_or_file).resolve()
    if src.is_dir():
        vocab_path = src / "qwen.tiktoken"
        if not vocab_path.exists():
            matches = list(src.rglob("*.tiktoken"))
            if not matches:
                raise FileNotFoundError(f"No .tiktoken file found in {src}")
            vocab_path = matches[0]
    else:
        vocab_path = src

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file missing: {vocab_path}")

    if target_bin_path is None:
        dest_bin = vocab_path.with_suffix(".ranks.bin")
    else:
        dest_bin = Path(target_bin_path).resolve()

    dest_bin.parent.mkdir(parents=True, exist_ok=True)
    compile_vocab(vocab_path, dest_bin)
    return dest_bin


class Encoding:
    def __init__(
        self,
        name: str = "YuE2",
        ranks_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self._name = name
        self._binding = get_binding()

        resolved_ranks: Optional[Path] = None
        if ranks_path is not None:
            p = Path(ranks_path).resolve()
            if p.is_file() and p.suffix == ".bin":
                resolved_ranks = p
            elif p.is_file() and p.suffix == ".tiktoken":
                bin_candidate = p.with_suffix(".ranks.bin")
                if not bin_candidate.exists():
                    compile_vocab(p, bin_candidate)
                resolved_ranks = bin_candidate
            elif p.is_dir():
                bin_candidate = p / "qwen.ranks.bin"
                if bin_candidate.exists():
                    resolved_ranks = bin_candidate
                else:
                    tik_candidate = p / "qwen.tiktoken"
                    if tik_candidate.exists():
                        resolved_ranks = compile_qwen_ranks_binary(tik_candidate, bin_candidate)

        if resolved_ranks is None or not resolved_ranks.exists():
            search_candidates = [
                ROOT_DIR / "qwen.ranks.bin",
                ROOT_DIR / "dist" / "bin" / "qwen.ranks.bin",
                ROOT_DIR.parent / "qwen.ranks.bin",
            ]
            for cand in search_candidates:
                if cand.is_file():
                    resolved_ranks = cand
                    break

        if resolved_ranks is None or not resolved_ranks.exists():
            tik_search = [
                ROOT_DIR / "qwen.tiktoken",
                ROOT_DIR.parent / "qwen.tiktoken",
            ]
            for tik_cand in tik_search:
                if tik_cand.is_file():
                    bin_dest = tik_cand.with_suffix(".ranks.bin")
                    if not bin_dest.exists():
                        compile_vocab(tik_cand, bin_dest)
                    resolved_ranks = bin_dest
                    break

        if resolved_ranks is None or not resolved_ranks.exists():
            raise FileNotFoundError(
                "Could not resolve qwen.ranks.bin or qwen.tiktoken. "
                "Provide an explicit ranks_path or ensure binaries are present in dist/bin/."
            )

        self._ranks_path = resolved_ranks
        self._ctx = self._binding.lib.tiktalkin_init(str(self._ranks_path).encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(f"Failed to initialize TikTalkin context from {self._ranks_path}")

        self._n_vocab = 184704

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._binding.lib.tiktalkin_destroy(self._ctx)
            self._ctx = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_vocab(self) -> int:
        return self._n_vocab

    def encode_ordinary(self, text: str) -> List[int]:
        if not text:
            return []
        raw = text.encode("utf-8")
        byte_len = len(raw)
        max_tokens = max(byte_len, 16)
        out_buf = (ctypes.c_int32 * max_tokens)()

        count = self._binding.lib.tiktalkin_encode_ordinary(
            self._ctx, raw, byte_len, out_buf, max_tokens
        )
        return out_buf[:count]

    def encode(
        self, text: str, allowed_special: Union[str, set] = "all"
    ) -> List[int]:
        if not text:
            return []
        raw = text.encode("utf-8")
        byte_len = len(raw)
        max_tokens = max(byte_len, 16)
        out_buf = (ctypes.c_int32 * max_tokens)()

        if allowed_special == "all":
            count = self._binding.lib.tiktalkin_encode(
                self._ctx, raw, byte_len, out_buf, max_tokens
            )
        else:
            count = self._binding.lib.tiktalkin_encode_ordinary(
                self._ctx, raw, byte_len, out_buf, max_tokens
            )
        return out_buf[:count]

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        if not tokens:
            return ""
        count = len(tokens)
        arr_type = ctypes.c_int32 * count
        c_tokens = arr_type(*tokens)

        alloc_size = max(count * 8 + 64, 1024)
        buf = ctypes.create_string_buffer(alloc_size)

        written = self._binding.lib.tiktalkin_decode(
            self._ctx, c_tokens, count, buf, alloc_size
        )
        if written >= alloc_size - 1:
            alloc_size = count * 32 + 4096
            buf = ctypes.create_string_buffer(alloc_size)
            written = self._binding.lib.tiktalkin_decode(
                self._ctx, c_tokens, count, buf, alloc_size
            )

        return buf.value.decode("utf-8", errors=errors)