#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fastGJK/fastGJK.cuh>
#include <iostream>

#include "common.cuh"

using namespace fastGJK;

static ConvexHull generateConvex(int nvrtx, val_t offsetX, val_t offsetY, val_t offsetZ)
{
    Vec3 *verts = new Vec3[nvrtx];

    val_t scaleX     = 0.5f + ((val_t)rand() / RAND_MAX) * 1.5f;
    val_t scaleY     = 0.5f + ((val_t)rand() / RAND_MAX) * 1.5f;
    val_t scaleZ     = 0.5f + ((val_t)rand() / RAND_MAX) * 1.5f;
    val_t baseRadius = 0.5f;

    for (int i = 0; i < nvrtx; i++) {
        val_t u     = ((val_t)rand() / RAND_MAX);
        val_t v     = ((val_t)rand() / RAND_MAX);
        val_t z     = 1.0f - 2.0f * u;
        val_t r     = sqrt(fmax(0.0f, 1.0f - z * z));
        val_t theta = 2.0f * M_PI * v;
        val_t vx    = baseRadius * r * cos(theta) * scaleX + offsetX;
        val_t vy    = baseRadius * r * sin(theta) * scaleY + offsetY;
        val_t vz    = baseRadius * z * scaleZ + offsetZ;
        verts[i]    = Vec3{vx, vy, vz};
    }

    return ConvexHull{.verts = verts, .n = nvrtx};
}

int main(int argc, const char *argv[])
{
    unsigned int n     = 10000;
    unsigned int nvrtx = 256;

    ConvexHull *collidersA = new ConvexHull[n];
    ConvexHull *collidersB = new ConvexHull[n];

    // Prepare data on the host-side
    auto generate_start = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n; ++i) {
        collidersA[i] = generateConvex(nvrtx, 0.0, 0.0, 0.0);
        collidersB[i] = generateConvex(nvrtx, 0.0, 0.0, 0.0);
    }
    auto generate_end = std::chrono::steady_clock::now();
    auto generate_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(generate_end - generate_start).count();
    std::cout << "Data generation time: " << generate_ms << " ms\n";

    // Move data to the device
    ConvexHull *collidersA_device, *collidersB_device;
    Vec3       *verts_device;
    Simplex    *simplices_device;
    val_t      *distances_device;
    auto copy_start = std::chrono::steady_clock::now();
    prepareDataOnDevice(
        collidersA, collidersB, collidersA_device, collidersB_device, verts_device, simplices_device, distances_device, n);
    auto copy_end = std::chrono::steady_clock::now();
    auto copy_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start).count();
    std::cout << "Copy time: " << copy_ms << " ms\n";

    // Warmup
    for (int i = 0; i < 10; ++i) {
        warp::gjk_process(collidersA_device, collidersB_device, n, simplices_device, distances_device);
    }
    cudaDeviceSynchronize();

    // Benchmark GJK
    constexpr int repeats = 100;
    auto gjk_start        = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        warp::gjk_process(collidersA_device, collidersB_device, n, simplices_device, distances_device);
    }
    cudaDeviceSynchronize();
    auto gjk_end = std::chrono::steady_clock::now();
    auto gjk_us  = std::chrono::duration_cast<std::chrono::microseconds>(gjk_end - gjk_start).count();
    std::cout << "GJK avg time: " << gjk_us / repeats << " us (" << repeats << " runs)\n";

    // Free memory
    freeDeviceMem(collidersA_device, collidersB_device, verts_device, simplices_device, distances_device);
    for (unsigned int i = 0; i < n; ++i) {
        delete[] collidersA[i].verts;
        delete[] collidersB[i].verts;
    }
    delete[] collidersA;
    delete[] collidersB;

    return 0;
}