// src/ui/panels/control.cpp
#include "pch.h"
#include "ui/panels/control.h"
#include "managers/scene.h"
#include "scenes/dynamic_scene.h"
#include "resources/model.h"

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

    ImGui::End();
}
}   // namespace CubeDemo::UI
