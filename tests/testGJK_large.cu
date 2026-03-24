#include <fastGJK/fastGJK.cuh>
#include <fstream>
#include <string>

#include "tool.cuh"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_SUITE("Test GJK with large dataset")
{
    TEST_CASE("Full dataset with 100000 samples")
    {
        const std::string filename = "data/input_100000.txt";
        unsigned int      n;
        int              *nvertsA, *nvertsB;

        // Determine data shape
        REQUIRE(readDatasetMetadata(filename, n, nvertsA, nvertsB));
        GJKState state(n, nvertsA, nvertsB);

        // Read dataset
        readDataset(filename, state);

        // Copy input data to GPU
        state.copyInputToGpu();

        // Run
        fastGJK::warp::gjk_process(
            state.hullsA.device, state.hullsB.device, state.n, state.simplices.device, state.distances.device);

        // Copy result to host
        state.copyOutputToCpu();

        // Verify results
        for (unsigned int i = 0; i < n; i++) {
            fastGJK::val_t actual   = state.distances.host[i];
            fastGJK::val_t expected = state.distances_expected[i];
            INFO("Pair index: ", i);
            INFO("Actual: ", actual, "; Expected: ", expected);
            INFO("CUDA error: ", std::string(cudaGetErrorString(cudaGetLastError())));
            fastGJK::val_t tol = std::max(fastGJK::val_t(0.01), std::abs(expected) * fastGJK::val_t(0.01));
            CHECK(std::abs(actual - expected) < tol);
        }
    }
}