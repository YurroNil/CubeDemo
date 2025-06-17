// include/graphics/bound_sphere.h
#pragma once

namespace CubeDemo {
class Mesh;
using MeshArray = std::vector<Mesh>;
// 包围球结构体
    struct BoundingSphere {
        vec3 Center; float Rad; // 包围球中心, 半径
        void Calc(const MeshArray& meshes); // 计算包围球
    };
}