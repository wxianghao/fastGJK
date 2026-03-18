#pragma once

namespace fastGJK {
// ============================================================================
// Macros and compile-time constants
// ============================================================================

// GJK config
#define GJK_MAX_ITERS 20

// Warp params
#define WARP_SIZE 32
#define WARP_MASK 0xFFFFFFFF

// Util macros
#define DEVICE_PREFIX __device__ __forceinline__

// Types config
#ifdef GJK_USE_DOUBLE
using val_t = double;
using vec_t = double4_32a;
constexpr val_t zero{0};
#else
using val_t = float;
using vec_t = float4;
constexpr val_t zero{0};
#endif


} // namespace fastGJK
