// src/scenes/mainScene.cpp
#include "scenes/mainScene.h"
#include "core/window.h"
#include "graphics/renderer.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
    extern bool DEBUG_LOD_MODE;
}

namespace CubeDemo::Scenes {

// MainScene实现
void MainScene::Init() {
    m_ShadowMap = Create.ShadowMap();
    m_ShadowShader = Create.ShadowShader();
    m_DirLight = Create.DirLight();
    isInited = true;
}

// 渲染深度到阴影贴图
void MainScene::RenderShadow(Camera* camera) {
    
    m_ShadowMap->BindForWriting();
    m_ShadowShader->Use();
    
    // 阴影矩阵计算
    const auto lightSpaceMatrix = m_ShadowMap->GetLightSpaceMatrix(m_DirLight);
    m_ShadowShader->SetMat4("lightSpaceMatrix", lightSpaceMatrix);

    // 简化绘制模型（仅位置属性）
    for (auto* model : MODEL_POINTERS) {
        if (model->IsReady() && camera->isSphereVisible(model->bounds.Center, model->bounds.Rad)) {
            m_ShadowShader->SetMat4("model", model->GetModelMatrix());
            model->DrawSimple();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // 传递阴影贴图
    m_ShadowMap->BindForReading(GL_TEXTURE1);
    MODEL_SHADER->SetInt("shadowMap", 1);
    MODEL_SHADER->SetMat4("lightSpaceMatrix", lightSpaceMatrix);
}

void MainScene::RenderMainPass(GLFWwindow* window, Camera* camera) {
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
    
    // 主着色器配置
    MODEL_SHADER->Use();
    m_ShadowMap->BindForReading(GL_TEXTURE1);

    // 摄像机参数传递
    MODEL_SHADER->ApplyCamera(*camera, Window::GetAspectRatio());
    
    // 模型绘制循环
    for (auto* model : MODEL_POINTERS) {
        if (!model->IsReady()) {
            std::cout << "[Render] 模型未就绪: " << model << std::endl;
            continue;
        }

        // 视椎体裁剪判断
        if (model->IsReady() &&
            camera->isSphereVisible(model->bounds.Center, model->bounds.Rad)
        ) {
            model->DrawCall(DEBUG_LOD_MODE, *MODEL_SHADER, camera->Position);
        }
    }
    /* ------应用光源着色器------ */
    MODEL_SHADER->SetDirLight(m_DirLight);
    MODEL_SHADER->SetViewPos(camera->Position);
}

void MainScene::Cleanup() {
    delete m_ShadowMap; m_ShadowMap = nullptr;
    delete m_ShadowShader; m_ShadowShader = nullptr;
    delete m_DirLight; m_DirLight = nullptr;
}

MainScene::~MainScene() {
    if(isCleanup == true) return;
    
    this->Cleanup();
    isCleanup = true;
}

// 乱七八糟的Getters
GSM* MainScene::GetShadowMap() const { return m_ShadowMap; }
Shader* MainScene::GetShadowShader() const { return m_ShadowShader; }
DL* MainScene::GetDirLight() const { return m_DirLight; }

}   // namespace CubeDemo::Scenes
