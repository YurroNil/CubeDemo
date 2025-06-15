// src/managers/model/cleanner.cpp
#include "pch.h"
#include "managers/model/cleanner.h"
#include "resources/model.h"
#include "graphics/shader.h"

namespace CubeDemo::Managers {
// 删除指定模型
void ModelCleanner::Delete(Model** model) {
    delete *model; *model = nullptr;
}
// 删除所有模型
void ModelCleanner::DeleteAll(std::vector<Model*> &models) {
    for(auto* &model : models) {
        delete model; model = nullptr;
    }
    models.clear();
}
// 删除着色器
void ModelCleanner::DeleteShader(Shader** shader) {
    delete *shader; *shader = nullptr;
}
void ModelCleanner::DeleteAllShader(std::vector<Shader*> &shaders) {
    for(auto* shader : shaders) {
        delete shader; shader = nullptr;
    }
    shaders.clear();
}
}   // namespace CubeDemo::Managers
