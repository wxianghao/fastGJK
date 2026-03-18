#include <fastGJK/fastGJK.cuh>
#include <iostream>

#include "common.cuh"

static fastGJK::ConvexHull createCube(fastGJK::val_t x, fastGJK::val_t y, fastGJK::val_t z, fastGJK::val_t halfExtent)
{
    using namespace fastGJK;
    ConvexHull hull;
    hull.n     = 8;
    hull.verts = new Vec3[8]{
        Vec3(x - halfExtent, y - halfExtent, z - halfExtent),
        Vec3(x + halfExtent, y - halfExtent, z - halfExtent),
        Vec3(x - halfExtent, y + halfExtent, z - halfExtent),
        Vec3(x + halfExtent, y + halfExtent, z - halfExtent),
        Vec3(x - halfExtent, y - halfExtent, z + halfExtent),
        Vec3(x + halfExtent, y - halfExtent, z + halfExtent),
        Vec3(x - halfExtent, y + halfExtent, z + halfExtent),
        Vec3(x + halfExtent, y + halfExtent, z + halfExtent),
    };
    return hull;
}

static fastGJK::ConvexHull createPoly(std::initializer_list<fastGJK::Vec3> verts)
{
    fastGJK::ConvexHull hull;
    hull.n     = static_cast<int>(verts.size());
    hull.verts = new fastGJK::Vec3[hull.n];
    int i      = 0;
    for (const auto &v : verts)
        hull.verts[i++] = v;
    return hull;
}

void printSimplex(const fastGJK::Simplex &simplex)
{
    printf("Simplex vertices: %d\n", simplex.n);
    for (int i = 0; i < simplex.n; ++i) {
        const fastGJK::SupportPoint support = simplex.supports[i];
        printf("\tSupport A: (%6.3f, %6.3f, %6.3f); Support B: (%6.3f %6.3f %6.3f); Support: (%6.3f, %6.3f, %6.3f)\n",
               support.vertA.x(),
               support.vertA.y(),
               support.vertA.z(),
               support.vertB.x(),
               support.vertB.y(),
               support.vertB.z(),
               support.supportPoint.x(),
               support.supportPoint.y(),
               support.supportPoint.z());
    }
}

void testCube()
{
    // Test two touching cubes
    // d=0
    {
        auto cubeA = createCube(0, 0, 0, 2);
        auto cubeB = createCube(2, 0, 0, 2);

        printf("Test two touching cubes.\n");

        fastGJK::ConvexHull *cubeA_device, *cubeB_device;
        fastGJK::Simplex    *simplex_device, *simplex_host;
        fastGJK::Vec3       *verts_device;
        fastGJK::val_t      *distance_device, *distance_host;
        prepareDataOnDevice(
            &cubeA, &cubeB, cubeA_device, cubeB_device, verts_device, simplex_device, distance_device, 1);
        fastGJK::warp::gjk_process(cubeA_device, cubeB_device, 1, simplex_device, distance_device);
        copyResultToHost(simplex_device, distance_device, simplex_host, distance_host, 1);

        // Print simplex
        printSimplex(*simplex_host);
        printf("GJK distance: %6.3f\n", *distance_host);
        printf("\n");

        // Free pointers
        freeDeviceMem(cubeA_device, cubeB_device, verts_device, simplex_device, distance_device);
        freeHostMem(simplex_host, distance_host);
        delete[] cubeA.verts;
        delete[] cubeB.verts;
    }

    // Test two separate cubes
    // d=1
    {
        auto cubeA = createCube(0, 0, 0, 2);
        auto cubeB = createCube(5, 0, 0, 2);

        printf("Test two separate cubes.\n");

        fastGJK::ConvexHull *cubeA_device, *cubeB_device;
        fastGJK::Simplex    *simplex_device, *simplex_host;
        fastGJK::Vec3       *verts_device;
        fastGJK::val_t      *distance_device, *distance_host;
        prepareDataOnDevice(
            &cubeA, &cubeB, cubeA_device, cubeB_device, verts_device, simplex_device, distance_device, 1);
        fastGJK::warp::gjk_process(cubeA_device, cubeB_device, 1, simplex_device, distance_device);
        copyResultToHost(simplex_device, distance_device, simplex_host, distance_host, 1);

        // Print result
        printSimplex(*simplex_host);
        printf("GJK distance: %6.3f\n", *distance_host);
        printf("\n");

        // Free pointers
        freeDeviceMem(cubeA_device, cubeB_device, verts_device, simplex_device, distance_device);
        freeHostMem(simplex_host, distance_host);
        delete[] cubeA.verts;
        delete[] cubeB.verts;
    }
}

void testPolyhedron()
{
    // Test two seperate polyhedrons
    {
        auto polyA = createPoly({{0.0, 0.0, 0.0},
                                 {-1.0, 1.0, 0.0},
                                 {-1.0, -1.0, 0.0},
                                 {-1.0, 0.0, 1.0},
                                 {-1.0, 0.0, -1.0},
                                 {-2.0, 1.0, 1.0},
                                 {-4.0, 2.0, 0.0},
                                 {-4.0, -2.0, 0.0},
                                 {-4.0, 0.0, 2.0},
                                 {-4.0, 0.0, -2.0}});
        auto polyB = createPoly({{3.1415, 0.0, 0.0},
                                 {5.1415, 1.0, -1.0},
                                 {5.1415, -1.0, 1.0},
                                 {5.1415, -1.0, -1.0},
                                 {11.1415, 2.0, 2.0},
                                 {11.1415, 2.0, -2.0},
                                 {11.1415, -2.0, 2.0},
                                 {11.1415, -2.0, -2.0},
                                 {12.1415, 0.0, 3.0}});

        printf("Test two seperate polyhedrons.\n");

        fastGJK::ConvexHull *polyA_device, *polyB_device;
        fastGJK::Simplex    *simplex_device, *simplex_host;
        fastGJK::Vec3       *verts_device;
        fastGJK::val_t      *distance_device, *distance_host;
        prepareDataOnDevice(
            &polyA, &polyB, polyA_device, polyB_device, verts_device, simplex_device, distance_device, 1);
        fastGJK::warp::gjk_process(polyA_device, polyB_device, 1, simplex_device, distance_device);
        copyResultToHost(simplex_device, distance_device, simplex_host, distance_host, 1);

        // Print result
        printSimplex(*simplex_host);
        printf("GJK distance: %6.3f\n", *distance_host);
        printf("\n");

        // Free pointers
        freeDeviceMem(polyA_device, polyB_device, verts_device, simplex_device, distance_device);
        freeHostMem(simplex_host, distance_host);
        delete[] polyA.verts;
        delete[] polyB.verts;
    }
}

int main()
{
    testCube();
    testPolyhedron();

    return 0;
}