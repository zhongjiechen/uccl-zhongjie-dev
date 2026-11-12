#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single-node, one-GPU-per-process BF16 collectives. The identifier must be
// identical at every rank and unique for each communicator lifetime.
void* uccl_lite_mp_create(const char* identifier, int ranks, int rank,
                          size_t max_bytes);
// operation: 1=AllGather, 2=ReduceScatter(SUM), 3=AllReduce(SUM). count is
// the per-rank input element count for AG, per-rank output element count for
// RS, and input/output element count for AR.
// Enqueues on the caller's CUDA stream and returns before completion.
int uccl_lite_mp_enqueue(void* handle, int operation, const void* input,
                         void* output, size_t count, void* stream);
int uccl_lite_mp_destroy(void* handle);
const char* uccl_lite_mp_last_error(void);

#ifdef __cplusplus
}
#endif
