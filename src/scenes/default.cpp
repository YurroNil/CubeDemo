// src/scenes/default.cpp
#include "pch.h"
#include "core/window.h"
#include "graphics/renderer.h"
#include "managers/sceneMng.h"
#include "managers/lightMng.h"
#include "prefabs/shadow_map.h"
#include "resources/model.h"
#include "utils/defines.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
    extern LightMng* LIGHT_MNG;
    extern SceneMng* SCENE_MNG;
}

namespace CubeDemo::Scenes {

DefaultScene::DefaultScene() {
    name = "默认场景";
    id = "default";
}

// DefaultScene实现
void DefaultScene::Init() {
    if(s_isInited) return;

    // 设置场景ID
    SCENE_MNG->Current = SceneID::DEFAULT;

    // 创建方向光
    m_DirLight = LIGHT_MNG->Create.DirLight();
    
    // 使用配置文件的数据来设置光源参数
    LightMng::SetLightsData(SCENE_CONF_PATH + string("default/lights.json"), m_DirLight);

    MODEL_SHADER->Use();
    MODEL_SHADER->SetBool("useDayLighting", true);
}

void DefaultScene::Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) {
    // 设置视口
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());

    shadow_map->BindForReading(GL_TEXTURE1);

    // 摄像机参数传递
    MODEL_SHADER->ApplyCamera(camera, Window::GetAspectRatio());

    // 模型绘制循环
    for (auto* model : MODEL_POINTERS) model->DrawCall(MODEL_SHADER, camera);

    /* ------应用光源着色器------ */
    m_DirLight->SetShader(*MODEL_SHADER);
    MODEL_SHADER->SetViewPos(camera->Position);
}

void DefaultScene::Cleanup() {
    if(!s_isInited || s_isCleanup) return;

    if(m_DirLight != nullptr) {
        delete m_DirLight; m_DirLight = nullptr;
    }
    s_isInited = false;
}

DefaultScene::~DefaultScene() {}

}   // namespace CubeDemo::Scenes
