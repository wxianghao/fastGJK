#pragma once

#include <limits>

#include "geom.cuh"

namespace fastGJK::warp {
// Tolerance constants
static constexpr val_t epsilon{std::numeric_limits<val_t>::epsilon()};
static constexpr val_t epsRel{epsilon * 1e4};
static constexpr val_t epsTot{epsilon * 1e2};
static constexpr val_t epsRel2{epsRel * epsRel};


// ============================================================================
// Helper functions
// ============================================================================

static DEVICE_PREFIX val_t max(val_t a, val_t b) { return a > b ? a : b; }

static DEVICE_PREFIX void support_search(const Vec3 *verts_in, int n, const Vec3 &dir_in, Vec3 &support_mut)
{
    unsigned int tid       = threadIdx.x;
    unsigned int warp_lane = tid % WARP_SIZE;

    val_t localMax    = support_mut.dot(dir_in);
    int   localMaxIdx = -1;

    // Reduce to each thread's local max
    for (unsigned int i = warp_lane; i < n; i += WARP_SIZE) {
        // *global -> register
        val_t s = verts_in[i].dot(dir_in);
        if (s > localMax) {
            localMax    = s;
            localMaxIdx = i;
        }
    }

    // Warp reduce to find the global max
    // * XOR shuffle ensures all the threads have the reduction result, saving one broadcast
    val_t globalMax    = localMax;
    int   globalMaxIdx = localMaxIdx;
    val_t partnerMax;
    int   partnerMaxIdx;
#pragma unroll
    for (unsigned int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        partnerMax    = __shfl_xor_sync(WARP_MASK, globalMax, mask);
        partnerMaxIdx = __shfl_xor_sync(WARP_MASK, globalMaxIdx, mask);
        if (partnerMax > globalMax || (partnerMax == globalMax && partnerMaxIdx != -1 && globalMaxIdx != -1)) {
            globalMax    = partnerMax;
            globalMaxIdx = partnerMaxIdx;
        }
    }

    // * Broadcast required: two vertex may have the same dot product with the direction
    globalMaxIdx = __shfl_sync(WARP_MASK, globalMaxIdx, 0);

    if (globalMaxIdx != -1) {
        support_mut = verts_in[globalMaxIdx];
    }
}

static DEVICE_PREFIX void handleSimplexLine(Simplex &simplex_mut, Vec3 &dir_out)
{
    const Vec3 &p = simplex_mut.supports[1].supportPoint;
    const Vec3 &q = simplex_mut.supports[0].supportPoint;

    if (hff1(p, q)) {
        dir_out = projectOriginOnLine(p, q);
    }
    else {
        simplex_mut.degenerate1D1(dir_out);
    }
}

static DEVICE_PREFIX void handleSimplexTriangle(Simplex &simplex_mut, Vec3 &dir_out)
{

    const Vec3 &p = simplex_mut.supports[2].supportPoint;
    const Vec3 &q = simplex_mut.supports[1].supportPoint;
    const Vec3 &r = simplex_mut.supports[0].supportPoint;

    bool hff1_pq = hff1(p, q);
    bool hff1_pr = hff1(p, r);

    if (hff1_pq) {
        if (!hff2(p, q, r)) {
            if (hff1_pr) {
                if (!hff2(p, r, q)) {
                    dir_out = projectOriginOnPlane(p, q, r);
                }
                else {
                    dir_out = projectOriginOnLine(p, r);
                    simplex_mut.degenerate2D13(dir_out);
                }
            }
            else {
                dir_out = projectOriginOnPlane(p, q, r);
            }
        }
        else {
            dir_out = projectOriginOnLine(p, q);
            simplex_mut.degenerate2D12(dir_out);
        }
    }
    else if (hff1_pr) {
        if (!hff2(p, r, q)) {
            dir_out = projectOriginOnPlane(p, q, r);
        }
        else {
            dir_out = projectOriginOnLine(p, r);
            simplex_mut.degenerate2D13(dir_out);
        }
    }
    else {
        simplex_mut.degenerate2D1(dir_out);
    }
}

static DEVICE_PREFIX void handleSimplexTetra(Simplex &simplex_mut, Vec3 &dir_out)
{
    // Save local copies to avoid aliasing during rearrangement
    const SupportPoint s1 = simplex_mut.supports[3]; // newest
    const SupportPoint s2 = simplex_mut.supports[2];
    const SupportPoint s3 = simplex_mut.supports[1];
    const SupportPoint s4 = simplex_mut.supports[0]; // oldest

    const Vec3 &s1p = s1.supportPoint;
    const Vec3 &s2p = s2.supportPoint;
    const Vec3 &s3p = s3.supportPoint;
    const Vec3 &s4p = s4.supportPoint;

    // Edge tests from s1
    bool hff1_s12 = hff1(s1p, s2p);
    bool hff1_s13 = hff1(s1p, s3p);
    bool hff1_s14 = hff1(s1p, s4p);
    int  dotTotal = (int)hff1_s12 + (int)hff1_s13 + (int)hff1_s14;

    // S3Dregion1: no edges face origin, reduce to just s1
    if (dotTotal == 0) {
        simplex_mut.n           = 1;
        simplex_mut.supports[0] = s1;
        dir_out                 = s1p;
        return;
    }

    // Face tests (hff3 for each face opposite a vertex)
    bool hff3_134 = hff3(s1p, s3p, s4p); // face opposite s2
    bool hff3_142 = hff3(s1p, s4p, s2p); // face opposite s3
    bool hff3_123 = hff3(s1p, s2p, s3p); // face opposite s4

    // Tetrahedron orientation
    Vec3 s1s3 = s3p - s1p;
    Vec3 s1s4 = s4p - s1p;
    Vec3 s1s2 = s2p - s1p;
    bool sss  = s1s3.dot(s1s4.cross(s1s2)) <= 0;

    // testPlane: 0 = face IS facing origin, 1 = face is NOT facing origin
    int testPlaneTwo   = (int)(hff3_134 != sss);
    int testPlaneThree = (int)(hff3_142 != sss);
    int testPlaneFour  = (int)(hff3_123 != sss);
    int planeSum       = testPlaneTwo + testPlaneThree + testPlaneFour;

    // Vertex array indexed by [0]=s4, [1]=s3, [2]=s2 (matching reference)
    const SupportPoint verts[3]      = {s4, s3, s2};
    const bool         hff1_tests[3] = {hff1_s14, hff1_s13, hff1_s12};

    int          i, j, k;
    SupportPoint si, sj, sk;

    switch (planeSum) {
    case 3:
        // Origin inside tetrahedron
        dir_out = Vec3{0, 0, 0};
        break;

    case 2: {
        // One face faces origin → reduce to triangle, call handleSimplexTriangle
        if (!testPlaneTwo) {
            // Remove s2: keep s1, s3, s4
            simplex_mut.n           = 3;
            simplex_mut.supports[2] = s1;
            // supports[1] = s3 (already there)
            // supports[0] = s4 (already there)
        }
        else if (!testPlaneThree) {
            // Remove s3: keep s1, s2, s4
            simplex_mut.n           = 3;
            simplex_mut.supports[2] = s1;
            simplex_mut.supports[1] = s2;
            // supports[0] = s4 (already there)
        }
        else {
            // Remove s4: keep s1, s2, s3
            simplex_mut.n           = 3;
            simplex_mut.supports[2] = s1;
            simplex_mut.supports[1] = s2;
            simplex_mut.supports[0] = s3;
        }
        handleSimplexTriangle(simplex_mut, dir_out);
        break;
    }

    case 1: {
        // Two faces face origin
        // k = vertex on the non-facing face (must be in solution)
        if (testPlaneTwo) {
            k = 2;
            i = 1;
            j = 0;
        }
        else if (testPlaneThree) {
            k = 1;
            i = 0;
            j = 2;
        }
        else {
            k = 0;
            i = 2;
            j = 1;
        }
        si              = verts[i];
        sj              = verts[j];
        sk              = verts[k];
        const Vec3 &sip = si.supportPoint;
        const Vec3 &sjp = sj.supportPoint;
        const Vec3 &skp = sk.supportPoint;

        if (dotTotal == 1) {
            if (hff1_tests[k]) {
                if (!hff2(s1p, skp, sip)) {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = si;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, sip, skp);
                }
                else if (!hff2(s1p, skp, sjp)) {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = sj;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, sjp, skp);
                }
                else {
                    simplex_mut.n           = 2;
                    simplex_mut.supports[1] = s1;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnLine(s1p, skp);
                }
            }
            else if (hff1_tests[i]) {
                if (!hff2(s1p, sip, skp)) {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = si;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, sip, skp);
                }
                else {
                    simplex_mut.n           = 2;
                    simplex_mut.supports[1] = s1;
                    simplex_mut.supports[0] = si;
                    dir_out                 = projectOriginOnLine(s1p, sip);
                }
            }
            else {
                if (!hff2(s1p, sjp, skp)) {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = sj;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, sjp, skp);
                }
                else {
                    simplex_mut.n           = 2;
                    simplex_mut.supports[1] = s1;
                    simplex_mut.supports[0] = sj;
                    dir_out                 = projectOriginOnLine(s1p, sjp);
                }
            }
        }
        else if (dotTotal == 2) {
            if (hff1_tests[i]) {
                if (!hff2(s1p, skp, sip)) {
                    if (!hff2(s1p, sip, skp)) {
                        simplex_mut.n           = 3;
                        simplex_mut.supports[2] = s1;
                        simplex_mut.supports[1] = si;
                        simplex_mut.supports[0] = sk;
                        dir_out                 = projectOriginOnPlane(s1p, sip, skp);
                    }
                    else {
                        simplex_mut.n           = 2;
                        simplex_mut.supports[1] = s1;
                        simplex_mut.supports[0] = sk;
                        dir_out                 = projectOriginOnLine(s1p, skp);
                    }
                }
                else {
                    if (!hff2(s1p, skp, sjp)) {
                        simplex_mut.n           = 3;
                        simplex_mut.supports[2] = s1;
                        simplex_mut.supports[1] = sj;
                        simplex_mut.supports[0] = sk;
                        dir_out                 = projectOriginOnPlane(s1p, sjp, skp);
                    }
                    else {
                        simplex_mut.n           = 2;
                        simplex_mut.supports[1] = s1;
                        simplex_mut.supports[0] = sk;
                        dir_out                 = projectOriginOnLine(s1p, skp);
                    }
                }
            }
            else if (hff1_tests[j]) {
                if (!hff2(s1p, skp, sjp)) {
                    if (!hff2(s1p, sjp, skp)) {
                        simplex_mut.n           = 3;
                        simplex_mut.supports[2] = s1;
                        simplex_mut.supports[1] = sj;
                        simplex_mut.supports[0] = sk;
                        dir_out                 = projectOriginOnPlane(s1p, sjp, skp);
                    }
                    else {
                        simplex_mut.n           = 2;
                        simplex_mut.supports[1] = s1;
                        simplex_mut.supports[0] = sj;
                        dir_out                 = projectOriginOnLine(s1p, sjp);
                    }
                }
                else {
                    if (!hff2(s1p, skp, sip)) {
                        simplex_mut.n           = 3;
                        simplex_mut.supports[2] = s1;
                        simplex_mut.supports[1] = si;
                        simplex_mut.supports[0] = sk;
                        dir_out                 = projectOriginOnPlane(s1p, sip, skp);
                    }
                    else {
                        simplex_mut.n           = 2;
                        simplex_mut.supports[1] = s1;
                        simplex_mut.supports[0] = sk;
                        dir_out                 = projectOriginOnLine(s1p, skp);
                    }
                }
            }
        }
        else { // dotTotal == 3
            bool hff2_ik = hff2(s1p, sip, skp);
            bool hff2_jk = hff2(s1p, sjp, skp);
            bool hff2_ki = hff2(s1p, skp, sip);
            bool hff2_kj = hff2(s1p, skp, sjp);

            if (hff2_ki && hff2_kj) {
                simplex_mut.n           = 2;
                simplex_mut.supports[1] = s1;
                simplex_mut.supports[0] = sk;
                dir_out                 = projectOriginOnLine(s1p, skp);
            }
            else if (hff2_ki) {
                if (hff2_jk) {
                    simplex_mut.n           = 2;
                    simplex_mut.supports[1] = s1;
                    simplex_mut.supports[0] = sj;
                    dir_out                 = projectOriginOnLine(s1p, sjp);
                }
                else {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = sj;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, skp, sjp);
                }
            }
            else {
                if (hff2_ik) {
                    simplex_mut.n           = 2;
                    simplex_mut.supports[1] = s1;
                    simplex_mut.supports[0] = si;
                    dir_out                 = projectOriginOnLine(s1p, sip);
                }
                else {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = si;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, skp, sip);
                }
            }
        }
        break;
    }

    case 0: {
        // All faces face origin
        if (dotTotal == 1) {
            // i is the vertex with positive hff1 test
            if (hff1_s13) {
                k = 2;
                i = 1;
                j = 0;
            }
            else if (hff1_s14) {
                k = 1;
                i = 0;
                j = 2;
            }
            else {
                k = 0;
                i = 2;
                j = 1;
            }
            si              = verts[i];
            sj              = verts[j];
            sk              = verts[k];
            const Vec3 &sip = si.supportPoint;
            const Vec3 &sjp = sj.supportPoint;
            const Vec3 &skp = sk.supportPoint;

            if (!hff2(s1p, sip, sjp)) {
                simplex_mut.n           = 3;
                simplex_mut.supports[2] = s1;
                simplex_mut.supports[1] = si;
                simplex_mut.supports[0] = sj;
                dir_out                 = projectOriginOnPlane(s1p, sip, sjp);
            }
            else if (!hff2(s1p, sip, skp)) {
                simplex_mut.n           = 3;
                simplex_mut.supports[2] = s1;
                simplex_mut.supports[1] = si;
                simplex_mut.supports[0] = sk;
                dir_out                 = projectOriginOnPlane(s1p, sip, skp);
            }
            else {
                simplex_mut.n           = 2;
                simplex_mut.supports[1] = s1;
                simplex_mut.supports[0] = si;
                dir_out                 = projectOriginOnLine(s1p, sip);
            }
        }
        else if (dotTotal == 2) {
            // i is the vertex with negative hff1 test
            if (!hff1_s13) {
                k = 2;
                i = 1;
                j = 0;
            }
            else if (!hff1_s14) {
                k = 1;
                i = 0;
                j = 2;
            }
            else {
                k = 0;
                i = 2;
                j = 1;
            }
            si              = verts[i];
            sj              = verts[j];
            sk              = verts[k];
            const Vec3 &sip = si.supportPoint;
            const Vec3 &sjp = sj.supportPoint;
            const Vec3 &skp = sk.supportPoint;

            if (!hff2(s1p, sjp, skp)) {
                if (!hff2(s1p, skp, sjp)) {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = sj;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, sjp, skp);
                }
                else if (!hff2(s1p, skp, sip)) {
                    simplex_mut.n           = 3;
                    simplex_mut.supports[2] = s1;
                    simplex_mut.supports[1] = si;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnPlane(s1p, skp, sip);
                }
                else {
                    simplex_mut.n           = 2;
                    simplex_mut.supports[1] = s1;
                    simplex_mut.supports[0] = sk;
                    dir_out                 = projectOriginOnLine(s1p, skp);
                }
            }
            else if (!hff2(s1p, sjp, sip)) {
                simplex_mut.n           = 3;
                simplex_mut.supports[2] = s1;
                simplex_mut.supports[1] = si;
                simplex_mut.supports[0] = sj;
                dir_out                 = projectOriginOnPlane(s1p, sip, sjp);
            }
            else {
                simplex_mut.n           = 2;
                simplex_mut.supports[1] = s1;
                simplex_mut.supports[0] = sj;
                dir_out                 = projectOriginOnLine(s1p, sjp);
            }
        }
        break;
    }

    default:
        break;
    }
}

static DEVICE_PREFIX void handleSimplex(Simplex &simplex_mut, Vec3 &dir_out)
{
    switch (simplex_mut.n) {
    case 2:
        handleSimplexLine(simplex_mut, dir_out);
        break;
    case 3:
        handleSimplexTriangle(simplex_mut, dir_out);
        break;
    case 4:
        handleSimplexTetra(simplex_mut, dir_out);
        break;
    default:
        break;
    }
}


// ============================================================================
// Device-level APIs
// ============================================================================

DEVICE_PREFIX
void gjk_process_device(const Vec3 *vertsA_in,
                        int         nA,
                        const Vec3 *vertsB_in,
                        int         nB,
                        Simplex    &simplex_out,
                        val_t      &distance_out)
{
    val_t norm2SupportMax{0};

    // Load first pair of vertices as the initial support
    // * global -> register
    Vec3 supportA{vertsA_in[0]};
    Vec3 supportB{vertsB_in[0]};

    // Initialize simplex
    simplex_out.push({supportA, supportB});

    // Initialize search direction
    Vec3 dir = simplex_out.supports[0].supportPoint;

    // GJK loop
    int k = 0;
    do {
        Vec3 dirMinus = -dir;

        // Search for support point
        support_search(vertsA_in, nA, dirMinus, supportA);
        support_search(vertsB_in, nB, dir, supportB);
        SupportPoint support{supportA, supportB};

        // Check exit condition 1
        val_t norm2Dir     = dir.norm2();
        val_t exceedTolRel = norm2Dir - dir.dot(support.supportPoint);
        if (exceedTolRel <= epsRel * norm2Dir || exceedTolRel < epsTot || norm2Dir < epsRel2) {
            break;
        }

        // Add new vertex to simplex
        simplex_out.push(support);

        // Handle simplex
        handleSimplex(simplex_out, dir);

        // Check exit condition 2
        if (simplex_out.n > 0)
            norm2SupportMax = max(norm2SupportMax, simplex_out.supports[0].supportPoint.norm2());
        if (simplex_out.n > 1)
            norm2SupportMax = max(norm2SupportMax, simplex_out.supports[1].supportPoint.norm2());
        if (simplex_out.n > 2)
            norm2SupportMax = max(norm2SupportMax, simplex_out.supports[2].supportPoint.norm2());
        if (simplex_out.n > 3)
            norm2SupportMax = max(norm2SupportMax, simplex_out.supports[3].supportPoint.norm2());
        norm2Dir = dir.norm2();
        if (norm2Dir <= (epsTot * epsTot * norm2SupportMax)) {
            break;
        }

        ++k;
    } while (k < GJK_MAX_ITERS);

    distance_out = sqrt(dir.norm2());
}

// ============================================================================
// Kernel-level APIs
// ============================================================================

__global__ void gjk_process_kernel(const ConvexHull *hullsA_in,
                                   const ConvexHull *hullsB_in,
                                   unsigned int      n,
                                   Simplex          *simplices_out,
                                   val_t            *distances_out)
{
    unsigned int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int collisionIdx = idx / WARP_SIZE;
    unsigned int warpLane     = threadIdx.x % WARP_SIZE;

    if (collisionIdx >= n) {
        return;
    }

    // * global -> register: broadcast of verts and n could be combined?
    const Vec3 *vertsA = hullsA_in[collisionIdx].verts;
    int         nA     = hullsA_in[collisionIdx].n;
    const Vec3 *vertsB = hullsB_in[collisionIdx].verts;
    int         nB     = hullsB_in[collisionIdx].n;
    Simplex     simplex;
    val_t       distance;


    // Call device function to handle each collision pair by a warp
    gjk_process_device(vertsA, nA, vertsB, nB, simplex, distance);

    if (warpLane == 0) {
        // * register -> global
        simplices_out[collisionIdx] = simplex;
        distances_out[collisionIdx] = distance;
    }
}

// ============================================================================
// Host-side APIs
// ============================================================================

template <unsigned int blockSize = 256>
void gjk_process(const ConvexHull *hullsA_in,
                 const ConvexHull *hullsB_in,
                 unsigned int      n,
                 Simplex          *simplices_out,
                 val_t            *distances_out)
{
    static_assert(blockSize % WARP_SIZE == 0, "block_size must be divisible by WARP_SIZE!");
    unsigned int collisionsPerBlock = blockSize / WARP_SIZE;
    unsigned int numBlocks          = (n + collisionsPerBlock - 1) / collisionsPerBlock;
    gjk_process_kernel<<<numBlocks, blockSize>>>(hullsA_in, hullsB_in, n, simplices_out, distances_out);
}

} // namespace fastGJK::warp
