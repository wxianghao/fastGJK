#pragma once

#include <algorithm>
#include <fastGJK/fastGJK.cuh>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>

#define AVAILABLE_BLOCK_SIZES 32, 64, 128

template <int First, int... Rest>
bool dispatch_gjk(int64_t              blockSize,
                  fastGJK::ConvexHull *hullsA,
                  fastGJK::ConvexHull *hullsB,
                  unsigned int         n,
                  fastGJK::Simplex    *simplices,
                  float               *distances)
{
    if (blockSize == First) {
        fastGJK::warp::gjk_process<First>(hullsA, hullsB, n, simplices, distances);
        return true;
    }
    if constexpr (sizeof...(Rest) > 0) {
        return dispatch_gjk<Rest...>(blockSize, hullsA, hullsB, n, simplices, distances);
    }
    return false;
}

template <typename T> struct GJKData
{
    unsigned int n;
    T           *host;
    T           *device;

    GJKData(unsigned int n)
        : n{n}
    {
        host = new T[n];
        cudaMalloc(&device, n * sizeof(T));
    }

    ~GJKData()
    {
        delete[] host;
        cudaFree(device);
    }

    void toGpu() { cudaMemcpy(device, host, n * sizeof(T), cudaMemcpyHostToDevice); }
    void toCpu() { cudaMemcpy(host, device, n * sizeof(T), cudaMemcpyDeviceToHost); }

    GJKData(const GJKData &)            = delete;
    GJKData &operator=(const GJKData &) = delete;
};

struct GJKConvexData
{
private:
    fastGJK::Vec3 *dverts_;
    fastGJK::Vec3 *hverts_;
    unsigned int   totalVerts;

public:
    unsigned int         n;
    fastGJK::ConvexHull *host;
    fastGJK::ConvexHull *device;
    GJKConvexData(unsigned int n, int nverts[])
        : n{n}
    {
        // Count total number of vertices
        totalVerts = 0;
        for (unsigned int i = 0; i < n; ++i) {
            totalVerts += nverts[i];
        }

        // Allocate memory
        host    = new fastGJK::ConvexHull[n];
        hverts_ = new fastGJK::Vec3[totalVerts];
        cudaMalloc(&device, n * sizeof(fastGJK::ConvexHull));
        cudaMalloc(&dverts_, totalVerts * sizeof(fastGJK::Vec3));

        // Record reference to contiguous vertex data and size information
        auto         tmp    = new fastGJK::ConvexHull[n];
        unsigned int offset = 0;
        for (unsigned int i = 0; i < n; ++i) {
            host[i].n     = nverts[i];
            host[i].verts = hverts_ + offset;
            tmp[i].n      = nverts[i];
            tmp[i].verts  = dverts_ + offset;
            offset += nverts[i];
        }
        cudaMemcpy(device, tmp, n * sizeof(fastGJK::ConvexHull), cudaMemcpyHostToDevice);

        delete[] tmp;
    }

    ~GJKConvexData()
    {
        delete[] host;
        delete[] hverts_;
        cudaFree(device);
        cudaFree(dverts_);
    }

    void toGpu() { cudaMemcpy(dverts_, hverts_, totalVerts * sizeof(fastGJK::Vec3), cudaMemcpyHostToDevice); }

    GJKConvexData(const GJKConvexData &)            = delete;
    GJKConvexData &operator=(const GJKConvexData &) = delete;
};

struct GJKState
{
private:
    int *nvertsA;
    int *nvertsB;

public:
    unsigned int              n;
    GJKConvexData             hullsA;
    GJKConvexData             hullsB;
    GJKData<fastGJK::val_t>   distances;
    fastGJK::val_t           *distances_expected;
    GJKData<fastGJK::Simplex> simplices;

    GJKState(unsigned int n, int *nvertsA, int *nvertsB)
        : n{n}
        , hullsA{n, nvertsA}
        , hullsB{n, nvertsB}
        , distances{n}
        , distances_expected{new fastGJK::val_t[n]}
        , simplices{n}
    {
        this->nvertsA = nvertsA;
        this->nvertsB = nvertsB;
    }

    void copyInputToGpu()
    {
        hullsA.toGpu();
        hullsB.toGpu();
    }

    void copyOutputToCpu()
    {
        distances.toCpu();
        simplices.toCpu();
    }

    ~GJKState()
    {
        delete[] distances_expected;
        delete[] nvertsA;
        delete[] nvertsB;
    }
};

bool readDatasetMetadata(const std::string &filename, unsigned int &n, int *&nvertsA, int *&nvertsB)
{
    std::ifstream  in{filename};
    fastGJK::val_t t;

    // Ensure the file is open
    if (!in.is_open()) {
        return false;
    }

    // Read the number of samples
    in >> n;
    nvertsA = new int[n];
    nvertsB = new int[n];

    // Read each sample
    for (unsigned int i = 0; i < n; ++i) {
        // Read distance
        in >> t;
        // Read first object
        in >> nvertsA[i];
        for (int j = 0; j < nvertsA[i]; ++j) {
            fastGJK::val_t x, y, z;
            in >> x >> y >> z;
        }
        // Read second object
        in >> nvertsB[i];
        for (int j = 0; j < nvertsB[i]; ++j) {
            fastGJK::val_t x, y, z;
            in >> x >> y >> z;
        }
    }

    in.close();
    return true;
}

bool readDataset(const std::string &filename, GJKState &state)
{
    std::ifstream in{filename};
    unsigned int  n;

    // Ensure the file is open
    if (!in.is_open()) {
        return false;
    }

    // Read the number of samples
    in >> n;

    // Read each sample
    for (unsigned int i = 0; i < n; ++i) {
        auto &A = state.hullsA.host[i];
        auto &B = state.hullsB.host[i];

        // Read distance
        in >> state.distances_expected[i];

        // Read first object
        in >> A.n;
        for (int j = 0; j < A.n; ++j) {
            fastGJK::val_t x, y, z;
            in >> x >> y >> z;
            A.verts[j] = {x, y, z};
        }

        // Read second object
        in >> B.n;
        for (int j = 0; j < B.n; ++j) {
            fastGJK::val_t x, y, z;
            in >> x >> y >> z;
            B.verts[j] = {x, y, z};
        }
    }

    in.close();
    return true;
}