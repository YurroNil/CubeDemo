// src/resources/model.cpp

#include "resources/model.h"
#include "kits/glfw.h"

using ML = CubeDemo::Loaders::Model;
namespace CubeDemo {

// Model构造函数传递路径到Loaders:Model类
Model::Model(const string& path) : Loaders::Model(path) {}

// 根据不同模式使用对应的绘制指令
void Model::DrawCall(bool mode, Shader& shader, const vec3& camera_pos) {
    if(mode) LodDraw(shader, camera_pos);
    else NormalDraw(shader);
}

// 普通模式绘制模型
void Model::NormalDraw(Shader& shader) {
    shader.SetMat4("model", GetModelMatrix());

    if(ML::isLoading().load()) return; // 加载中不绘制

    for (const Mesh& mesh : GetMeshes()) {
        mesh.Draw(shader);
    }
}

// LOD模式绘制模型 (使用LOD系统绘制模型)
void Model::LodDraw(Shader& shader, const vec3& camera_pos) {
    shader.SetMat4("model", GetModelMatrix());

    const Graphics::LODLevel& level = GetLODSystem().SelectLevel(bounds.Center, camera_pos);

    for (const Mesh& mesh : level.GetMeshes()) {
        mesh.Draw(shader);
    }
}

void Model::DrawSimple() const {
    for (const auto& mesh : GetMeshes()) {
        glBindVertexArray(mesh.GetVAO());
        glDrawElements(GL_TRIANGLES, mesh.GetIndexCount(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}
}   // namespace CubeDemo
