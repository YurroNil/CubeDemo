// src/resources/model.cpp
#include "pch.h"
#include "resources/model.h"
#include "graphics/shader.h"
#include "loaders/model.h"

using ML = CubeDemo::Loaders::Model;

namespace CubeDemo {

// Model构造函数传递路径到Loaders:Model类
Model::Model(const string& path)
    : m_Path(path), Managers::ModelGetter(this) {}

// 绘制函数. 判断是否符合条件，若符合条件则调用NormalDraw来更进一步地绘制网格
void Model::DrawCall(Camera* camera) {
    
    // 模型绘制循环
    if (!IsReady()) {
        std::cout << "[Render] 模型未就绪: " << this << std::endl;
        return;
    }
    // 视椎体裁剪判断
    // if (IsReady() && camera->isSphereVisible(bounds.Center, bounds.Rad)) { ... }
    NormalDraw(camera->Position);
}

// 普通模式绘制模型
void Model::NormalDraw(const vec3& camera_pos) {

    ModelShader->SetMat4("model", m_ModelMatrix);
    if(m_isLoading.load() || m_Meshes.empty() ) return; // 加载中不绘制
    // 绘制该模型中的所有网格
    for (const Mesh& mesh : m_Meshes) mesh.Draw(ModelShader);
}

void Model::DrawSimple() const {
    for (const auto& mesh : m_Meshes) {
        glBindVertexArray(mesh.GetVAO());
        glDrawElements(GL_TRIANGLES, mesh.GetIndexCount(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}

const void Model::UpdateModelMatrix() {
    m_ModelMatrix = mat4(1.0f);
    m_ModelMatrix = translate(m_ModelMatrix, m_Position);
    m_ModelMatrix = rotate(m_ModelMatrix, m_Rotation, vec3(0.0f, 1.0f, 0.0f));
    m_ModelMatrix = scale(m_ModelMatrix, m_Scale);
}

Model::~Model() {
    // 删除着色器
    std::cout << "  删除" << m_ID << "的着色器: " << ModelShader << std::endl;
    delete ModelShader;
    ModelShader = nullptr;
}
}   // namespace CubeDemo
