// src/resources/model.cpp
#include "pch.h"
#include "resources/model.h"
#include "graphics/shader.h"

using ML = CubeDemo::Loaders::Model;
namespace CubeDemo {

// Model构造函数传递路径到Loaders:Model类
Model::Model(const string& path) : Loaders::Model(path) {}

// 根据不同模式使用对应的绘制指令
void Model::DrawCall(Shader& shader, const vec3& camera_pos) {
    NormalDraw(shader);
}

// 普通模式绘制模型
void Model::NormalDraw(Shader& shader) {
    shader.SetMat4("model", GetModelMatrix());

    if(ML::isLoading().load()) return; // 加载中不绘制

    for (const Mesh& mesh : GetMeshes()) {
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

/* ------------ 清理器 ------------ */

// 删除指定模型
void Model::Delete(Model** model) {
    delete *model; *model = nullptr;
}
// 删除所有模型
void Model::DeleteAll(std::vector<Model*> &models) {
    for(auto* &model : models) {
        delete model; model = nullptr;
    }
    models.clear();
}
// 删除着色器
void Model::DeleteShader(Shader** shader) {
    delete *shader; *shader = nullptr;
}
void Model::DeleteAllShader(std::vector<Shader*> &shaders) {
    for(auto* shader : shaders) {
        delete shader; shader = nullptr;
    }
    shaders.clear();
}

}   // namespace CubeDemo
