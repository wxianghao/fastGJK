#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <fastGJK/fastGJK.cuh>
#include <fstream>

void prepareDataOnDevice(const fastGJK::ConvexHull *hullsA_host,
                         const fastGJK::ConvexHull *hullsB_host,
                         fastGJK::ConvexHull      *&hullsA_device,
                         fastGJK::ConvexHull      *&hullsB_device,
                         fastGJK::Vec3    *&verts_device, // Contiguous memory chunk for storing all the vertices
                         fastGJK::Simplex *&simplices_device,
                         fastGJK::val_t   *&distances_device,
                         unsigned int       n)
{
    // Count the total number of vertices
    unsigned int totalVerts = 0u;
    for (unsigned int i = 0; i < n; ++i) {
        totalVerts += hullsA_host[i].n;
        totalVerts += hullsB_host[i].n;
    }

    // Allocate memory on device
    cudaMalloc(&hullsA_device, n * sizeof(fastGJK::ConvexHull));
    cudaMalloc(&hullsB_device, n * sizeof(fastGJK::ConvexHull));
    cudaMalloc(&verts_device, totalVerts * sizeof(fastGJK::Vec3));
    cudaMalloc(&simplices_device, n * sizeof(fastGJK::Simplex));
    cudaMalloc(&distances_device, n * sizeof(fastGJK::val_t));

    // Temporary ConvexHulls for transferring device pointers
    auto hullsA_tmp = new fastGJK::ConvexHull[n];
    auto hullsB_tmp = new fastGJK::ConvexHull[n];

    // Copy vertices and assign vertex pointers
    unsigned int offset = 0;
    // hullsA
    for (unsigned int i = 0; i < n; ++i) {
        int nverts = hullsA_host[i].n;
        cudaMemcpy(verts_device + offset, hullsA_host[i].verts, nverts * sizeof(fastGJK::Vec3), cudaMemcpyHostToDevice);
        hullsA_tmp[i].n     = nverts;
        hullsA_tmp[i].verts = verts_device + offset;
        offset += nverts;
    }
    // hullsB
    for (unsigned int i = 0; i < n; ++i) {
        int nverts = hullsB_host[i].n;
        cudaMemcpy(verts_device + offset, hullsB_host[i].verts, nverts * sizeof(fastGJK::Vec3), cudaMemcpyHostToDevice);
        hullsB_tmp[i].n     = nverts;
        hullsB_tmp[i].verts = verts_device + offset;
        offset += nverts;
    }

    // Copy convex hulls to device
    cudaMemcpy(hullsA_device, hullsA_tmp, n * sizeof(fastGJK::ConvexHull), cudaMemcpyHostToDevice);
    cudaMemcpy(hullsB_device, hullsB_tmp, n * sizeof(fastGJK::ConvexHull), cudaMemcpyHostToDevice);

    assert(offset == totalVerts);

    // Free temporary data
    delete[] hullsA_tmp;
    delete[] hullsB_tmp;
}

void freeDeviceMem(fastGJK::ConvexHull *hullsA_device,
                   fastGJK::ConvexHull *hullsB_device,
                   fastGJK::Vec3       *verts_device,
                   fastGJK::Simplex    *simplices_device,
                   fastGJK::val_t      *distances_device)
{
    cudaFree(hullsA_device);
    cudaFree(hullsB_device);
    cudaFree(verts_device);
    cudaFree(simplices_device);
    cudaFree(distances_device);
}

void copyResultToHost(fastGJK::Simplex  *simplices_device,
                      fastGJK::val_t    *distances_device,
                      fastGJK::Simplex *&simplices_host,
                      fastGJK::val_t   *&distances_host,
                      unsigned int       n)
{
    // Allocate memory on host
    simplices_host = (fastGJK::Simplex *)malloc(n * sizeof(fastGJK::Simplex));
    distances_host = (fastGJK::val_t *)malloc(n * sizeof(fastGJK::val_t));

    // Copy data to host
    cudaMemcpy(simplices_host, simplices_device, n * sizeof(fastGJK::Simplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances_host, distances_device, n * sizeof(fastGJK::val_t), cudaMemcpyDeviceToHost);
}

void freeHostMem(fastGJK::Simplex *simplices_host, fastGJK::val_t *distances_host)
{
    free(simplices_host);
    free(distances_host);
}

bool readDataset(const std::string    &filename,
                 fastGJK::ConvexHull *&hullsA,
                 fastGJK::ConvexHull *&hullsB,
                 fastGJK::val_t      *&distances,
                 unsigned int         &n)
{
    std::ifstream in{filename};

    // Ensure the file is open
    if (!in.is_open()) {
        return false;
    }

    // Read the number of samples
    in >> n;

    // Allocate memory
    distances = new fastGJK::val_t[n];
    hullsA    = new fastGJK::ConvexHull[n];
    hullsB    = new fastGJK::ConvexHull[n];

    // Read each sample
    for (unsigned int i = 0; i < n; ++i) {
        auto &A = hullsA[i];
        auto &B = hullsB[i];

        // Read distance
        in >> distances[i];

        // Read first object
        in >> A.n;
        A.verts = new fastGJK::Vec3[A.n];
        for (int j = 0; j < A.n; ++j) {
            fastGJK::val_t x, y, z;
            in >> x >> y >> z;
            A.verts[j] = {x, y, z};
        }

        // Read second object
        in >> B.n;
        B.verts = new fastGJK::Vec3[B.n];
        for (int j = 0; j < B.n; ++j) {
            fastGJK::val_t x, y, z;
            in >> x >> y >> z;
            B.verts[j] = {x, y, z};
        }
    }

    in.close();
    return true;
}
