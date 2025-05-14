// src/resources/model.cpp

#include "resources/model.h"

using ML = CubeDemo::Loaders::Model;
namespace CubeDemo {

// Model构造函数传递路径到Loaders:Model类
Model::Model(const string& path) : Loaders::Model(path) {}

// 普通模式绘制模型
void Model::NormalDraw(Shader& shader) {
    shader.SetMat4("model", GetModelMatrix());

    if(ML::isLoading().load()) return; // 加载中不绘制

    for (const Mesh& mesh : GetMeshes()) {
        mesh.Draw(shader);
    }
}

// LOD模式绘制模型 (使用LOD系统绘制模型)
void Model::LodDraw(Shader& shader, const vec3& cameraPos) {
    shader.SetMat4("model", GetModelMatrix());

    const Graphics::LODLevel& level = GetLODSystem().SelectLevel(bounds.Center, cameraPos);

    for (const Mesh& mesh : level.GetMeshes()) {
        mesh.Draw(shader);
    }
}
}   // namespace CubeDemo
