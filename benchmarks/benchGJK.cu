#include <cstdio>
#include <fastGJK/fastGJK.cuh>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include <string>

#include "tool.cuh"

#define AVAILABLE_BLOCK_SIZES 32, 64, 128

std::string filename;

void bench(nvbench::state &nvstate)
{
    unsigned int n;
    int         *nvertsA, *nvertsB;

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
    nvstate.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        if (!dispatch_gjk<AVAILABLE_BLOCK_SIZES>(blockSize,
                                                 state.hullsA.device,
                                                 state.hullsB.device,
                                                 n,
                                                 state.simplices.device,
                                                 state.distances.device)) {
            printf("Unsupported block size: %ld\n", blockSize);
        }
        else {
            cudaDeviceSynchronize();
        }
    });
}

NVBENCH_BENCH(bench).add_int64_axis("block_size", {AVAILABLE_BLOCK_SIZES});

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "./benchGJK <input_file> [<nvbench_args>]\n";
        std::exit(1);
    }

    filename = argv[1];
    std::cout << "Benchmarking with " << filename << "\n";

    // Filter nvbench arguments
    std::vector<char *> nvbench_args;
    nvbench_args.reserve(argc - 1);
    for (int i = 0; i < argc; ++i) {
        if (i != 1) {
            nvbench_args.push_back(argv[i]);
        }
    }

    NVBENCH_MAIN_BODY(nvbench_args.size(), nvbench_args.data());
    return 0;
}