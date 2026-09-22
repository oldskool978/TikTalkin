#ifndef TIKTALKIN_H
#define TIKTALKIN_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#if defined(TTKN_BUILD_DLL)
#define TTKN_API __declspec(dllexport)
#else
#define TTKN_API __declspec(dllimport)
#endif
#else
#define TTKN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tiktalkin_ctx_t tiktalkin_ctx_t;

#pragma pack(push, 1)
typedef struct {
    int32_t  id;
    uint16_t length;
    char     literal[58];
} ttkn_special_disk_t;
#pragma pack(pop)

typedef struct {
    uint32_t total_tokens;
    uint32_t table_capacity;
    uint32_t sso_tokens;
    uint32_t max_psl;
    uint64_t string_blob_bytes;
    double   load_factor;
    double   sso_ratio;
    double   mean_psl;
    double   psl_variance;
    double   compile_seconds;
} ttkn_telemetry_t;

TTKN_API tiktalkin_ctx_t *tiktalkin_init(const char *ranks_bin_path);

TTKN_API int32_t tiktalkin_encode(
    tiktalkin_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
);

TTKN_API int32_t tiktalkin_encode_ordinary(
    tiktalkin_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
);

TTKN_API int32_t tiktalkin_decode(
    tiktalkin_ctx_t *ctx,
    const int32_t *token_ids,
    uint32_t id_count,
    char *out_text,
    uint32_t max_bytes
);

TTKN_API int32_t tiktalkin_compile_vocab_with_telemetry(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *out_telemetry_json,
    ttkn_telemetry_t *out_telemetry
);

TTKN_API int32_t tiktalkin_compile_vocab(
    const char *in_tiktoken_path,
    const char *out_bin_path
);

TTKN_API int32_t tiktalkin_compile_vocab_v4(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *regex_pattern,
    const ttkn_special_disk_t *specials,
    uint32_t specials_count,
    uint32_t eot_token_id,
    const char *out_telemetry_json,
    ttkn_telemetry_t *out_telemetry
);

TTKN_API void tiktalkin_destroy(tiktalkin_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif