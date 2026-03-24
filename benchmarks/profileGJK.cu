#include <cstdio>
#include <cuda_profiler_api.h>
#include <fastGJK/fastGJK.cuh>
#include <iostream>
#include <string>

#include "tool.cuh"

#ifdef FASTGJK_LARGE_DATASET
const std::string filename = "./data/input_100000.txt";
#else
const std::string filename = "./data/input_5000.txt";
#endif

int main()
{
    unsigned int n;
    int         *nvertsA, *nvertsB;

    // Determine data shape
    readDatasetMetadata(filename, n, nvertsA, nvertsB);
    GJKState state(n, nvertsA, nvertsB);

    // Read dataset
    readDataset(filename, state);

    // Copy input data to GPU
    state.copyInputToGpu();

    // Warmup
    for (int i = 0; i < 8; ++i) {
        fastGJK::warp::gjk_process<128>(
            state.hullsA.device, state.hullsB.device, state.n, state.simplices.device, state.distances.device);
        fastGJK::warp::gjk_process<256>(
            state.hullsA.device, state.hullsB.device, state.n, state.simplices.device, state.distances.device);
        fastGJK::warp::gjk_process<512>(
            state.hullsA.device, state.hullsB.device, state.n, state.simplices.device, state.distances.device);
    }

    // Profile
    cudaProfilerStart();
    fastGJK::warp::gjk_process<128>(
        state.hullsA.device, state.hullsB.device, state.n, state.simplices.device, state.distances.device);
    fastGJK::warp::gjk_process<256>(
        state.hullsA.device, state.hullsB.device, state.n, state.simplices.device, state.distances.device);
    fastGJK::warp::gjk_process<512>(
        state.hullsA.device, state.hullsB.device, state.n, state.simplices.device, state.distances.device);
    cudaProfilerStop();

    return 0;
}