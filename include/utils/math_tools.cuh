// include/utils/math_tools.cuh
#ifndef MATH_TOOLS_CUH
#define MATH_TOOLS_CUH

static __forceinline__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static __forceinline__ __device__ float3 normalize(const float3& v) {
    float len = sqrtf(dot(v, v));
    if (len > 0.0f) {
        float invLen = 1.0f / len;
        return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
    }
    return v;
}

static __forceinline__ __device__ float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

static __forceinline__ __device__ float3 operator*(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

static __forceinline__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator-(const float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}

static __forceinline__ __device__ float3 operator/(const float3& v, float s) {
    float invS = 1.0f / s;
    return make_float3(v.x * invS, v.y * invS, v.z * invS);
}

static __forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __device__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

#ifndef M_PI
#define M_PI CUDART_PI_F
#endif  // M_PI

#endif  // MATH_TOOLS_CUH
