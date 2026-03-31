#include <cstdio>
#include <fstream>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include <openGJK_GPU.h>
#include <string>
#include <vector>

std::string filename;

struct OpenGJKState
{
    unsigned int n;

    // Host data
    std::vector<gkPolytope> h_bd1, h_bd2;
    std::vector<gkSimplex>  h_simplices;
    std::vector<gkFloat>    h_distances;

    // Flattened vertex storage (host)
    std::vector<gkFloat> h_coords1_flat, h_coords2_flat;

    // Device pointers
    gkPolytope *d_bd1 = nullptr, *d_bd2 = nullptr;
    gkFloat    *d_coord1 = nullptr, *d_coord2 = nullptr;
    gkSimplex  *d_simplices = nullptr;
    gkFloat    *d_distances = nullptr;
};

bool readDataset(const std::string &filename, OpenGJKState &state)
{
    std::ifstream in{filename};
    if (!in.is_open())
        return false;

    unsigned int n;
    in >> n;
    state.n = n;

    // First pass: read all data and compute total vertex counts
    struct PairData
    {
        gkFloat              dist;
        int                  nA;
        std::vector<gkFloat> coordsA; // flattened [x0,y0,z0,x1,y1,z1,...]
        int                  nB;
        std::vector<gkFloat> coordsB;
    };
    std::vector<PairData> pairs(n);

    unsigned int totalVertsA = 0, totalVertsB = 0;
    for (unsigned int i = 0; i < n; ++i) {
        auto &p = pairs[i];
        in >> p.dist;

        in >> p.nA;
        p.coordsA.resize(p.nA * 3);
        for (int j = 0; j < p.nA; ++j) {
            in >> p.coordsA[j * 3 + 0] >> p.coordsA[j * 3 + 1] >> p.coordsA[j * 3 + 2];
        }
        totalVertsA += p.nA;

        in >> p.nB;
        p.coordsB.resize(p.nB * 3);
        for (int j = 0; j < p.nB; ++j) {
            in >> p.coordsB[j * 3 + 0] >> p.coordsB[j * 3 + 1] >> p.coordsB[j * 3 + 2];
        }
        totalVertsB += p.nB;
    }

    // Build flattened coordinate arrays and polytope structs
    state.h_coords1_flat.resize(totalVertsA * 3);
    state.h_coords2_flat.resize(totalVertsB * 3);
    state.h_bd1.resize(n);
    state.h_bd2.resize(n);
    state.h_simplices.resize(n);
    state.h_distances.resize(n);

    unsigned int offsetA = 0, offsetB = 0;
    for (unsigned int i = 0; i < n; ++i) {
        auto &p = pairs[i];

        // Copy A coords into flat array
        std::copy(p.coordsA.begin(), p.coordsA.end(), state.h_coords1_flat.begin() + offsetA * 3);
        state.h_bd1[i].numpoints = p.nA;
        state.h_bd1[i].s[0] = state.h_bd1[i].s[1] = state.h_bd1[i].s[2] = 0;
        state.h_bd1[i].s_idx                                            = 0;
        // coord pointer will be set after device allocation; store offset for now
        state.h_bd1[i].coord = reinterpret_cast<gkFloat *>((uintptr_t)(offsetA * 3));
        offsetA += p.nA;

        // Copy B coords into flat array
        std::copy(p.coordsB.begin(), p.coordsB.end(), state.h_coords2_flat.begin() + offsetB * 3);
        state.h_bd2[i].numpoints = p.nB;
        state.h_bd2[i].s[0] = state.h_bd2[i].s[1] = state.h_bd2[i].s[2] = 0;
        state.h_bd2[i].s_idx                                            = 0;
        state.h_bd2[i].coord = reinterpret_cast<gkFloat *>((uintptr_t)(offsetB * 3));
        offsetB += p.nB;

        state.h_simplices[i].nvrtx = 0;
    }

    return true;
}

void copyToDevice(OpenGJKState &state)
{
    const unsigned int n = state.n;

    // Allocate device coordinate arrays
    cudaMalloc(&state.d_coord1, state.h_coords1_flat.size() * sizeof(gkFloat));
    cudaMalloc(&state.d_coord2, state.h_coords2_flat.size() * sizeof(gkFloat));
    cudaMemcpy(state.d_coord1,
               state.h_coords1_flat.data(),
               state.h_coords1_flat.size() * sizeof(gkFloat),
               cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_coord2,
               state.h_coords2_flat.data(),
               state.h_coords2_flat.size() * sizeof(gkFloat),
               cudaMemcpyHostToDevice);

    // Patch coord pointers to device addresses
    std::vector<gkPolytope> tmp_bd1(state.h_bd1), tmp_bd2(state.h_bd2);
    for (unsigned int i = 0; i < n; ++i) {
        uintptr_t off1   = reinterpret_cast<uintptr_t>(tmp_bd1[i].coord);
        tmp_bd1[i].coord = state.d_coord1 + off1;

        uintptr_t off2   = reinterpret_cast<uintptr_t>(tmp_bd2[i].coord);
        tmp_bd2[i].coord = state.d_coord2 + off2;
    }

    // Allocate and copy polytope structs to device
    cudaMalloc(&state.d_bd1, n * sizeof(gkPolytope));
    cudaMalloc(&state.d_bd2, n * sizeof(gkPolytope));
    cudaMemcpy(state.d_bd1, tmp_bd1.data(), n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_bd2, tmp_bd2.data(), n * sizeof(gkPolytope), cudaMemcpyHostToDevice);

    // Allocate output arrays
    cudaMalloc(&state.d_simplices, n * sizeof(gkSimplex));
    cudaMalloc(&state.d_distances, n * sizeof(gkFloat));
}

void freeDevice(OpenGJKState &state)
{
    cudaFree(state.d_bd1);
    cudaFree(state.d_bd2);
    cudaFree(state.d_coord1);
    cudaFree(state.d_coord2);
    cudaFree(state.d_simplices);
    cudaFree(state.d_distances);
}

void bench(nvbench::state &nvstate)
{
    OpenGJKState state;
    readDataset(filename, state);
    copyToDevice(state);

    const unsigned int n = state.n;

    nvstate.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        compute_minimum_distance_device(n, state.d_bd1, state.d_bd2, state.d_simplices, state.d_distances);
    });

    freeDevice(state);
}

NVBENCH_BENCH(bench);

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "./benchOpenGJK <input_file> [<nvbench_args>]\n";
        std::exit(1);
    }

    filename = argv[1];
    std::cout << "Benchmarking OpenGJK-GPU with " << filename << "\n";

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
