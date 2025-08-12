// src/graphics/bound_sphere.cpp
#include "pch.h"
#include "graphics/bound_sphere.h"

namespace CubeDemo {

// 计算包围球
void BoundingSphere::Calc(const MeshArray& meshes) {
    using namespace glm;
    if (meshes.empty()) {
        Center = vec3(0.0f);
        Rad = 0.0f;
        return;
    }

    // 精确计算包围球（使用所有顶点）
    vec3 min_vert(FLT_MAX), max_vert(-FLT_MAX);
    for (const auto& mesh : meshes) {
        for (const auto& v : mesh.Vertices) {
            min_vert = min(min_vert, v.Position);
            max_vert = max(max_vert, v.Position);
        }
    }
    Center = (min_vert + max_vert) * 0.5f;
    Rad = 0.0f;

    // 计算最大距离
    for (const auto& mesh : meshes) {
        for (const auto& v : mesh.Vertices) {
            if (const float dist = length(v.Position - Center); dist > Rad) Rad = dist;
        }
    }
}
}