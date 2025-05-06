
#include "graphics/boundingSphere.h"

namespace CubeDemo {

// 计算包围球
void BoundingSphere::Calc(const MeshArray& meshes) {
    if (meshes.empty()) {
        Center = vec3(0.0f);
        Rad = 0.0f;
        return;
    }
    // 计算AABB
    vec3 minVert(FLT_MAX), maxVert(-FLT_MAX);
    for (const auto& mesh : meshes) {
        for (const auto& vert : mesh.Vertices) {
            minVert = glm::min(minVert, vert.Position);
            maxVert = glm::max(maxVert, vert.Position);
        }
    }
    // 中心点计算
    Center = (minVert + maxVert) * 0.5f;
    
    // 计算最大半径
    float maxDist = 0.0f;
    for (const auto& mesh : meshes) {
        for (const auto& vert : mesh.Vertices) {
            maxDist = glm::max(maxDist, glm::length(vert.Position - Center));
        }
    }
    Rad = maxDist;
}


}