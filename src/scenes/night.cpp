// src/scenes/night.cpp
#include "pch.h"
#include "scenes/night_inc.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
    extern LightMng* LIGHT_MNG;
    extern SceneMng* SCENE_MNG;
}

namespace CubeDemo::Scenes {

// 场景初始化
void NightScene::Init() {
    if(s_isInited) return;
    
    // 创建月光（方向光）
    m_MoonLight = LIGHT_MNG->Create.DirLight();
    // 创建聚光（光束）
    m_SpotLight = LIGHT_MNG->Create.SpotLight();

    // 将指针储存进LIGHT_MNG->Get成员中
    LIGHT_MNG->Get.SetDirLight(m_MoonLight).SetSpotLight(m_SpotLight);

    // 使用配置文件的数据来设置光源参数
    LightMng::SetLightsData(SCENE_CONF_PATH + string("night/lights.json"), m_SpotLight, m_MoonLight);

    // 加载体积光着色器
    Beam.CreateVolumShader();

    // 创建光束几何体
    Beam.CreateLightCone(
        Beam.GetEffects().radius,
        Beam.GetEffects().height
    );

    LightMng::SetLightsData(SCENE_CONF_PATH + string("night/lights.json"), &Beam);

    // 设置场景ID
    SCENE_MNG->Current = SceneID::NIGHT;
}

// 场景清理
void NightScene::Cleanup() {
    if(!s_isInited || s_isCleanup) return;

    // 删除光束
    Shader* shader = Beam.GetVolumShader();
    Beam.Remove.VolumShader().LightCone().NoiseTexture();
    // 删除定向光与聚光
    LIGHT_MNG->Remove.DirLight().SpotLight();
}

// 渲染场景
void NightScene::Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) {
    
    // 视口设置
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
    // 主着色器配置
    MODEL_SHADER->Use();

    shadow_map->BindForReading(GL_TEXTURE1);

    // 摄像机参数传递
    MODEL_SHADER->ApplyCamera(camera, Window::GetAspectRatio());

    // 模型绘制循环
    for (auto* model : MODEL_POINTERS) model->DrawCall(MODEL_SHADER, camera);

    MODEL_SHADER->SetViewPos(camera->Position);

    // 设置月光
    m_MoonLight->SetShader(*MODEL_SHADER);

    // 设置聚光着色器
    m_SpotLight->SetShader(*MODEL_SHADER);
    
    // 渲染体积光
    Beam.Render(camera, m_SpotLight);
}

NightScene::NightScene() {}
NightScene::~NightScene() {}

} // namespace
