#include <cstdio>
#include <cuda_profiler_api.h>
#include <fastGJK/fastGJK.cuh>
#include <iostream>
#include <string>
#include <vector>

#include "tool.cuh"

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "./profileGJK <input_file>\n";
        std::exit(1);
    }

    std::string filename = argv[1];
    std::cout << "Profiling with " << filename << "\n";

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
        for (auto &blockSize : std::vector<unsigned int>{AVAILABLE_BLOCK_SIZES}) {
            dispatch_gjk<AVAILABLE_BLOCK_SIZES>(blockSize,
                                                state.hullsA.device,
                                                state.hullsB.device,
                                                state.n,
                                                state.simplices.device,
                                                state.distances.device);
        }
    }

    // Profile
    cudaProfilerStart();
    for (auto &blockSize : std::vector<unsigned int>{AVAILABLE_BLOCK_SIZES}) {
        dispatch_gjk<AVAILABLE_BLOCK_SIZES>(blockSize,
                                            state.hullsA.device,
                                            state.hullsB.device,
                                            state.n,
                                            state.simplices.device,
                                            state.distances.device);
    }
    cudaProfilerStop();

    return 0;
}