// src/scenes/default.cpp
#include "pch.h"
#include "core/window.h"
#include "graphics/renderer.h"
#include "scenes/sceneMng.h"
#include "resources/model.h"
#include "core/camera.h"
#include "prefabs/light.h"
#include "prefabs/shadowMap.h"
#include "resources/model.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
    extern bool DEBUG_LOD_MODE;
}

namespace CubeDemo::Scenes {

// DefaultScene实现
void DefaultScene::Init(SceneMng* scene_inst, Light& light) {
    if(s_isInited) return;

    scene_inst->Current = SceneID::DEFAULT;

    // 创建平行光源
    light.Get.SetDirLight(light.Create.DirLight());
}

void DefaultScene::Render(
    GLFWwindow* window,
    Camera* camera,
    const Light& light,
    ShadowMap* shadow_map)
{
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
    // 主着色器配置
    MODEL_SHADER->Use();

    shadow_map->BindForReading(GL_TEXTURE1);

    // 摄像机参数传递
    MODEL_SHADER->ApplyCamera(camera, Window::GetAspectRatio());

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
    MODEL_SHADER->SetDirLight("dirLight", light.Get.DirLight());
    MODEL_SHADER->SetViewPos(camera->Position);

}

void DefaultScene::Cleanup(Light& light) {
    if(!s_isInited || s_isCleanup) return;

    light.Remove.DirLight();
}

DefaultScene::~DefaultScene() {
}
}   // namespace CubeDemo::Scenes
