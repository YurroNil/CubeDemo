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
    if(m_isInited) return;

    // 创建模型与着色器
    MIL::InitModels();

    // 创建方向光
    m_DirLight = LIGHT_MNG->Create.DirLight();
    // 创建点光
    m_SkyLight = LIGHT_MNG->Create.SkyLight();
    
    // 使用配置文件的数据来设置光源参数
    LightMng::SetLightsData(SCENE_CONF_PATH + string(id + "/lights.json"), m_DirLight, m_SkyLight);

    // 设置天空颜色为天蓝色
    glClearColor(0.5f, 0.7f, 1.0f, 1.0f);

}

void DefaultScene::Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) {
    // 设置视口
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());

    shadow_map->BindForReading(GL_TEXTURE1);

    // 使用着色器
    MODEL_MNG->AllUseShader(camera, Window::GetAspectRatio(), m_DirLight, nullptr, nullptr, m_SkyLight);

    // 模型绘制循环
    if(MODEL_POINTERS.empty()) return;
    for (auto* model : MODEL_POINTERS) model->DrawCall(camera);
}

void DefaultScene::Cleanup() {
    if(!m_isInited || m_isCleanup) return;

    // 删除光源
    if(m_DirLight != nullptr) { delete m_DirLight; m_DirLight = nullptr; }
    if(m_SkyLight != nullptr) { delete m_SkyLight; m_SkyLight = nullptr; }

    m_isInited = false; m_isCleanup = true;
}

DefaultScene::~DefaultScene() {}

}   // namespace CubeDemo::Scenes
