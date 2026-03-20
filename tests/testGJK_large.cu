#include <fastGJK/fastGJK.cuh>
#include <fstream>
#include <string>

#include "common.cuh"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_SUITE("Test GJK with large dataset")
{
    TEST_CASE("Full dataset with 100000 samples")
    {
        fastGJK::ConvexHull *hullsA_host, *hullsB_host;
        fastGJK::val_t      *distances_host, *expected;
        fastGJK::Simplex    *simplices_host;
        unsigned int         n;

        // Load data to main memory
        readDataset("data/input_100000.txt", hullsA_host, hullsB_host, expected, n);

        // Copy data to device memory
        fastGJK::ConvexHull *hullsA_device, *hullsB_device;
        fastGJK::val_t      *distances_device;
        fastGJK::Simplex    *simplices_device;
        fastGJK::Vec3       *verts_device;
        prepareDataOnDevice(hullsA_host,
                            hullsB_host,
                            hullsA_device,
                            hullsB_device,
                            verts_device,
                            simplices_device,
                            distances_device,
                            n);

        // Run
        fastGJK::warp::gjk_process(hullsA_device, hullsB_device, n, simplices_device, distances_device);

        // Copy result to host
        copyResultToHost(simplices_device, distances_device, simplices_host, distances_host, n);


        // Verify results
        for (unsigned int i = 0; i < n; i++) {
            INFO("Pair index: ", i);
            INFO("Actual: ", distances_host[i], "; Expected: ", expected[i]);
            fastGJK::val_t tol = std::max(fastGJK::val_t(0.01), std::abs(expected[i]) * fastGJK::val_t(0.01));
            CHECK(std::abs(distances_host[i] - expected[i]) < tol);
        }

        // Free resources
        freeDeviceMem(hullsA_device, hullsB_device, verts_device, simplices_device, distances_device);
        freeHostMem(simplices_host, distances_host);
        for (unsigned int i = 0; i < n; ++i) {
            delete[] hullsA_host[i].verts;
            delete[] hullsB_host[i].verts;
        }
        delete[] hullsA_host;
        delete[] hullsB_host;
        delete[] expected;
    }
}