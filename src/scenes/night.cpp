// src/scenes/night.cpp
#include "pch.h"
#include "scenes/night_inc.h"

// 外部变量声明
namespace CubeDemo {
    extern std::vector<Model*> MODEL_POINTERS;
    // 管理器
    extern ModelMng* MODEL_MNG;
    extern LightMng* LIGHT_MNG;
    extern SceneMng* SCENE_MNG;
}

// 别名
using MIL = CubeDemo::Loaders::ModelIniter;

namespace CubeDemo::Scenes {

NightScene::NightScene() {
    name = "夜晚场景";
    id = "night";
    Beam = VolumBeam();
}

// 场景初始化
void NightScene::Init() {
    if(s_isInited) return;
    
    // 创建模型与着色器
    MIL::InitModels();

    // 创建月光（方向光）
    m_MoonLight = LIGHT_MNG->Create.DirLight();
    // 创建聚光（光束）
    m_SpotLight = LIGHT_MNG->Create.SpotLight();

    // 使用配置文件的数据来设置光源参数
    LightMng::SetLightsData(SCENE_CONF_PATH + string("night/lights.json"), m_SpotLight, m_MoonLight);

    // 加载体积光着色器
    Beam.CreateVolumShader();

    // 创建光束几何体
    Beam.CreateLightCone(
        Beam.Effects.radius,
        Beam.Effects.height
    );
    
    // 使用配置文件的数据来设置光源参数
    LightMng::SetLightsData(SCENE_CONF_PATH + string("night/lights.json"), &Beam);
}

// 场景清理
void NightScene::Cleanup() {
    if(!s_isInited || s_isCleanup) return;

    // 删除光束着色器
    if(Beam.VolumShader != nullptr) {
        delete Beam.VolumShader; Beam.VolumShader = nullptr;
    }
    // 删除光束
    if(Beam.LightVolume != nullptr) {
        delete Beam.LightVolume; Beam.LightVolume = nullptr;
    }
    // 删除噪声纹理 (因为是智能指针，因此不需要delete)
    if(Beam.NoiseTexture != nullptr) Beam.NoiseTexture = nullptr;

    // 删除定向光与聚光
    if(m_MoonLight != nullptr) {
        delete m_MoonLight; m_MoonLight = nullptr;
    }
    if(m_SpotLight != nullptr) {
        delete m_SpotLight; m_SpotLight = nullptr;
    }
    // 删除模型与着色器
    MODEL_MNG->RmvAllShaders(); MODEL_MNG->RmvAllModels();

    s_isInited = false;
}

// 渲染场景
void NightScene::Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) {
    
    // 视口设置
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
    // 主着色器配置

    shadow_map->BindForReading(GL_TEXTURE1);

    MODEL_MNG->AllUseShader(camera, Window::GetAspectRatio(), m_MoonLight, m_SpotLight, nullptr);

    // 模型绘制循环
    for (auto* model : MODEL_POINTERS) model->DrawCall(camera);
    
    // 渲染体积光
    Beam.Render(camera, m_SpotLight);
}

NightScene::~NightScene() {}

} // namespace
