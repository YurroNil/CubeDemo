// src/ui/panels/control.cpp
#include "pch.h"
#include "ui/panels/control.h"
#include "managers/scene/mng.h"
#include "scenes/dynamic_scene.h"
#include "resources/model.h"
#include "graphics/ray_tracing.h"

namespace CubeDemo {
    extern Managers::SceneMng* SCENE_MNG;
    extern std::vector<Model*> MODEL_POINTERS;
    extern bool RT_DEBUG;
}

namespace CubeDemo::UI {
    
// 渲染控制面板
void ControlPanel::Render(Camera* camera) {
    ImGui::Begin("控制面板");

    // 添加一个滑动条，用于调整相机移动速度
    ImGui::SliderFloat("移动速度", &camera->attribute.movementSpeed, 1.0f, 30.0f);
    
    // 添加光线追踪开关
    static bool enableRT = false;
    if (ImGui::Checkbox("启用光线追踪", &enableRT)) {
        Renderer::RayTracingEnabled(enableRT);
    }

    // 光追调试信息
    if (!RT_DEBUG) { ImGui::End(); return; }


    ImGui::End();
}
}   // namespace CubeDemo::UI
