// src/scenes/default.cpp
#include "pch.h"
#include "scenes/default_inc.h"

// 外部变量声明
namespace CubeDemo {
    extern std::vector<Model*> MODEL_POINTERS;

    extern LightMng* LIGHT_MNG;
    extern SceneMng* SCENE_MNG;
    extern ModelMng* MODEL_MNG;
}

// 别名
using MIL = CubeDemo::Loaders::ModelIniter;

namespace CubeDemo::Scenes {

DefaultScene::DefaultScene() {
    name = "默认场景";
    id = "default";
}

// DefaultScene实现
void DefaultScene::Init() {
    if(s_isInited) return;

    // 创建模型与着色器
    MIL::InitModels();

    // 创建方向光
    m_DirLight = LIGHT_MNG->Create.DirLight();
    
    // 使用配置文件的数据来设置光源参数
    LightMng::SetLightsData(SCENE_CONF_PATH + string("default/lights.json"), m_DirLight);

}

void DefaultScene::Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) {
    // 设置视口
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());

    shadow_map->BindForReading(GL_TEXTURE1);

    // 使用着色器
    MODEL_MNG->AllUseShader(camera, Window::GetAspectRatio(), m_DirLight, nullptr, nullptr);

    // MODEL_SHADER->SetBool("useDayLighting", true);

    // 模型绘制循环
    for (auto* model : MODEL_POINTERS) model->DrawCall(camera);
}

void DefaultScene::Cleanup() {
    if(!s_isInited || s_isCleanup) return;

    // 删除光源
    if(m_DirLight != nullptr) {
        delete m_DirLight; m_DirLight = nullptr;
    }
    // 删除模型与着色器
    MODEL_MNG->RmvAllShaders(); MODEL_MNG->RmvAllModels();

    s_isInited = false;
}

DefaultScene::~DefaultScene() {}

}   // namespace CubeDemo::Scenes
