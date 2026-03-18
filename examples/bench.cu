#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fastGJK/fastGJK.cuh>
#include <iostream>

using namespace fastGJK;

static ConvexHull generateConvex(int nvrtx, val_t offsetX, val_t offsetY, val_t offsetZ)
{
    Vec3 *verts;
    cudaMallocHost(&verts, nvrtx * sizeof(Vec3));

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

static void copy_colliders_to_device(ConvexHull      *&colliders_device,
                                     Vec3            *&vrtx_device,
                                     const ConvexHull *colliders_host,
                                     unsigned int      n)
{
    // Count total number of vertices
    unsigned int total_nvrtx = 0;
    for (unsigned int i = 0; i < n; ++i) {
        total_nvrtx += colliders_host[i].n;
    }

    // Allocate memory
    cudaMalloc(&colliders_device, n * sizeof(ConvexHull));
    cudaMalloc(&vrtx_device, total_nvrtx * sizeof(Vec3));
    ConvexHull *colliders_tmp = new ConvexHull[n];

    unsigned int offset = 0;
    for (unsigned int i = 0; i < n; ++i) {
        // Copy vertices
        int nvrtx = colliders_host[i].n;
        cudaMemcpy(vrtx_device + offset, colliders_host[i].verts, nvrtx * sizeof(Vec3), cudaMemcpyHostToDevice);
        // Copy convex to host tmp
        colliders_tmp[i] = {.verts = vrtx_device + offset, .n = nvrtx};
        offset += nvrtx;
    }

    // Copy all the convexes
    cudaMemcpy(colliders_device, colliders_tmp, n * sizeof(ConvexHull), cudaMemcpyHostToDevice);

    // Clean-up memories
    delete[] colliders_tmp;
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
    Simplex    *simplices_device;
    ConvexHull *collidersA_device, *collidersB_device;
    Vec3       *vrtxA_device, *vrtxB_device;
    cudaMalloc(&simplices_device, n * sizeof(Simplex));
    auto copy_start = std::chrono::steady_clock::now();
    copy_colliders_to_device(collidersA_device, vrtxA_device, collidersA, n);
    copy_colliders_to_device(collidersB_device, vrtxB_device, collidersB, n);
    auto copy_end = std::chrono::steady_clock::now();
    auto copy_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start).count();
    std::cout << "Copy time: " << copy_ms << " ms\n";

    // Verify GJK
    auto gjk_start = std::chrono::steady_clock::now();
    warp::gjk_process(collidersA_device, collidersB_device, n, simplices_device);
    cudaDeviceSynchronize();
    auto gjk_end = std::chrono::steady_clock::now();
    auto gjk_us  = std::chrono::duration_cast<std::chrono::microseconds>(gjk_end - gjk_start).count();
    std::cout << "GJK time: " << gjk_us << " us\n";

    return 0;
}