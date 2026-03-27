#include <cstdio>
#include <fastGJK/fastGJK.cuh>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include <string>

#include "tool.cuh"

#ifdef FASTGJK_LARGE_DATASET
const std::string filename = "./data/input_100000.txt";
#else
const std::string filename = "./data/input_5000.txt";
#endif


void bench(nvbench::state &nvstate)
{
    unsigned int n;
    int         *nvertsA, *nvertsB;

    printf("Benchmarking GJK with %s\n", filename.c_str());

    // Query bench inputs
    const auto blockSize = nvstate.get_int64("block_size");

    // Determine data shape
    readDatasetMetadata(filename, n, nvertsA, nvertsB);
    GJKState state(n, nvertsA, nvertsB);

    // Read dataset
    readDataset(filename, state);

    // Copy input data to GPU
    state.copyInputToGpu();

    // Bench
    nvstate.exec([&](nvbench::launch &launch) {
        // fastGJK::warp::gjk_process_kernel<<<gridSize, blockSize, 0, launch.get_stream()>>>(
        //     state.hullsA.device, state.hullsB.device, n, state.simplices.device, state.distances.device);
        switch (blockSize) {
        case 64:
            fastGJK::warp::gjk_process<64>(
                state.hullsA.device, state.hullsB.device, n, state.simplices.device, state.distances.device);
            break;
        case 128:
            fastGJK::warp::gjk_process<128>(
                state.hullsA.device, state.hullsB.device, n, state.simplices.device, state.distances.device);
            break;
        case 256:
            fastGJK::warp::gjk_process<256>(
                state.hullsA.device, state.hullsB.device, n, state.simplices.device, state.distances.device);
            break;
        case 512:
            fastGJK::warp::gjk_process<512>(
                state.hullsA.device, state.hullsB.device, n, state.simplices.device, state.distances.device);
            break;
        case 1024:
            fastGJK::warp::gjk_process<1024>(
                state.hullsA.device, state.hullsB.device, n, state.simplices.device, state.distances.device);
            break;
        default:
            printf("Unsupported block size: %ld\n", blockSize);
            break;
        }
    });
}

NVBENCH_BENCH(bench).add_int64_axis("block_size", {64, 128, 256, 512, 1024});