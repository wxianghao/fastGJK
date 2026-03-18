#pragma once

#include <cstdint>

#include "comm.cuh"

namespace fastGJK {

// ============================================================================
// Data structures
// ============================================================================

struct Vec3
{
private:
    vec_t _data;

public:
    Vec3() = default;
    DEVICE_PREFIX Vec3(val_t x, val_t y, val_t z)
    {
        _data.x = x;
        _data.y = y;
        _data.z = z;
    }
    DEVICE_PREFIX val_t x() const { return _data.x; }
    DEVICE_PREFIX val_t y() const { return _data.y; }
    DEVICE_PREFIX val_t z() const { return _data.z; }
    DEVICE_PREFIX val_t dot(const Vec3 &rhs) const
    {
        return _data.x * rhs._data.x + _data.y * rhs._data.y + _data.z * rhs._data.z;
    }
    DEVICE_PREFIX val_t norm2() const { return _data.x * _data.x + _data.y * _data.y + _data.z * _data.z; }
    DEVICE_PREFIX Vec3  operator-(const Vec3 &rhs) const
    {
        val_t vx = _data.x - rhs._data.x;
        val_t vy = _data.y - rhs._data.y;
        val_t vz = _data.z - rhs._data.z;
        return Vec3{vx, vy, vz};
    }
    DEVICE_PREFIX Vec3 cross(const Vec3 &rhs) const
    {
        return Vec3{_data.y * rhs._data.z - _data.z * rhs._data.y,
                    _data.z * rhs._data.x - _data.x * rhs._data.z,
                    _data.x * rhs._data.y - _data.y * rhs._data.x};
    }
    DEVICE_PREFIX Vec3        operator-() const { return Vec3{-_data.x, -_data.y, -_data.z}; }
    DEVICE_PREFIX friend Vec3 operator*(val_t s, const Vec3 &v)
    {
        return Vec3{s * v._data.x, s * v._data.y, s * v._data.z};
    }
};

struct ConvexHull
{
    Vec3 *verts;
    int   n;
};

struct SupportPoint
{
    SupportPoint() = default;
    DEVICE_PREFIX SupportPoint(const Vec3 &a, const Vec3 &b)
        : vertA{a}
        , vertB{b}
        , supportPoint{a - b}
    {
    }
    Vec3 vertA;
    Vec3 vertB;
    Vec3 supportPoint;
};

struct Simplex
{
    SupportPoint supports[4];
    uint8_t      n;

    DEVICE_PREFIX Simplex()
        : n{0} {};

    DEVICE_PREFIX void push(const SupportPoint &support)
    {
        supports[n] = support;
        ++n;
    }

    DEVICE_PREFIX void degenerate1D1(Vec3 &dir_out)
    {
        n           = 1;
        supports[0] = supports[1];
        dir_out     = supports[0].supportPoint;
    }

    DEVICE_PREFIX void degenerate2D1(Vec3 &dir_out)
    {
        n           = 1;
        supports[0] = supports[2];
        dir_out     = supports[0].supportPoint;
    }

    DEVICE_PREFIX void degenerate2D12(Vec3 &dir_out)
    {
        n           = 2;
        supports[0] = supports[2];
    }

    DEVICE_PREFIX void degenerate2D13(Vec3 &dir_out)
    {
        n           = 2;
        supports[1] = supports[2];
    }
};

// ============================================================================
// Geometric functions
// ============================================================================

DEVICE_PREFIX Vec3 projectOriginOnLine(const Vec3 &p, const Vec3 &q)
{
    Vec3  pq = p - q;
    val_t t  = p.dot(pq) / pq.norm2();
    return p - t * pq;
}


DEVICE_PREFIX Vec3 projectOriginOnPlane(const Vec3 &p, const Vec3 &q, const Vec3 &r)
{
    Vec3  n = (p - q).cross(p - r);
    val_t t = p.dot(n) / n.norm2();
    return t * n;
}

DEVICE_PREFIX bool hff1(const Vec3 &p, const Vec3 &q) { return p.dot(p - q) > 0; }

DEVICE_PREFIX bool hff2(const Vec3 &p, const Vec3 &q, const Vec3 &r)
{
    Vec3 pq = p - q;
    Vec3 n  = pq.cross(p - r);
    Vec3 t  = pq.cross(p);
    return n.dot(t) < 0;
}

DEVICE_PREFIX bool hff3(const Vec3 &p, const Vec3 &q, const Vec3 &r)
{
    Vec3 n = (p - q).cross(p - r);
    return n.dot(p) <= 0;
}

} // namespace fastGJK
