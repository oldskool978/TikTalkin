#include "tiktalkin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

int main(int argc, char **argv) {
#if defined(_WIN32)
    int wargc = 0;
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        char **utf8_argv = (char **)malloc(sizeof(char *) * wargc);
        for (int i = 0; i < wargc; i++) {
            int needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
            utf8_argv[i] = (char *)malloc(needed);
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, utf8_argv[i], needed, NULL, NULL);
        }
        argv = utf8_argv;
        argc = wargc;
    }
#endif

    if (argc < 2) {
        printf("TikTalkin Native Sovereign Tokenizer CLI\n");
        printf("Usage:\n");
        printf("  tiktalkin-cli compile <qwen.tiktoken> <qwen.ranks.bin> [telemetry.json]\n");
        printf("  tiktalkin-cli bench <qwen.ranks.bin> <benchmark_text>\n");
        return 0;
    }

    if (strcmp(argv[1], "compile") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: Missing vocabulary input/output arguments.\n");
            return 1;
        }
        const char *in_tiktoken = argv[2];
        const char *out_bin = argv[3];
        const char *out_telem = (argc >= 5) ? argv[4] : NULL;

        ttkn_telemetry_t telem;
        int rc = tiktalkin_compile_vocab_with_telemetry(in_tiktoken, out_bin, out_telem, &telem);
        if (rc != 0) {
            fprintf(stderr, "[!] Serialization error with code: %d\n", rc);
            return 1;
        }
        printf("[+] Serialization Succeeded:\n");
        printf("    Tokens:         %u\n", telem.total_tokens);
        printf("    Capacity:       %u (Load: %.2f%%)\n", telem.table_capacity, telem.load_factor * 100.0);
        printf("    SSO Ratio:      %.2f%% (%u inline)\n", telem.sso_ratio * 100.0, telem.sso_tokens);
        printf("    Heap Blob:      %llu bytes\n", (unsigned long long)telem.string_blob_bytes);
        printf("    Mean PSL:       %.4f (Max: %u, Var: %.4f)\n", telem.mean_psl, telem.max_psl, telem.psl_variance);
        printf("    Duration:       %.4fs\n", telem.compile_seconds);
        return 0;
    }

    if (strcmp(argv[1], "bench") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: Missing benchmark binary and string arguments.\n");
            return 1;
        }
        tiktalkin_ctx_t *ctx = tiktalkin_init(argv[2]);
        if (!ctx) {
            fprintf(stderr, "[!] Failed to map ranks container: %s\n", argv[2]);
            return 1;
        }
        const char *test_prompt = argv[3];
        uint32_t in_len = (uint32_t)strlen(test_prompt);
        int32_t tokens[8192];
        clock_t t0 = clock();
        int32_t count = tiktalkin_encode(ctx, test_prompt, in_len, tokens, 8192);
        clock_t t1 = clock();
        double ms = ((double)(t1 - t0) / CLOCKS_PER_SEC) * 1000.0;
        printf("[+] Encoded %d tokens (%u bytes) in %.4f ms (%.2f MB/s)\n",
               count, in_len, ms, ((double)in_len / (1024.0 * 1024.0)) / (ms / 1000.0));

        uint32_t dec_alloc = in_len * 4 + 1024;
        char *decoded = (char *)malloc(dec_alloc);
        if (decoded) {
            tiktalkin_decode(ctx, tokens, (uint32_t)count, decoded, dec_alloc);
            printf("[+] Roundtrip Parity: %s\n", (strcmp(test_prompt, decoded) == 0) ? "EXACT MATCH" : "DIVERGENCE DETECTED");
            free(decoded);
        }
        tiktalkin_destroy(ctx);
        return 0;
    }

    return 0;
}