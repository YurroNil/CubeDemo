// src/ui/panels/control.cpp
#include "pch.h"
#include "ui/panels/control.h"

namespace CubeDemo::UI {
    
// 渲染控制面板
void ControlPanel::Render(Camera* camera) {
    ImGui::Begin("控制面板"); // 开始一个新的ImGui窗口，标题为"Control Panel"
    ImGui::SliderFloat("移动速度", &camera->attribute.movementSpeed, 1.0f, 30.0f); // 添加一个滑动条，用于调整相机移动速度
    
    if (ImGui::Button("全屏")) { // 添加一个按钮，用于切换全屏模式
        Window::ToggleFullscreen(Window::GetWindow());
    }
    ImGui::End(); // 结束ImGui窗口
}
}   // namespace CubeDemo::UI
