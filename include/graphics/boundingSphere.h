
#include "loaders/material.h"

#pragma once
namespace CubeDemo {

// 包围球结构体
    struct BoundingSphere {
        vec3 Center; float Rad; // 包围球中心, 半径
        void Calc(const MeshArray& meshes); // 计算包围球
    };
}