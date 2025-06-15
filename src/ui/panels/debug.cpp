// src/ui/panels/debug.cpp
#include "pch.h"
#include "ui/panels/debug.h"
#include "core/monitor.h"
#include "core/time.h"

namespace CubeDemo::UI {
// 渲染调试面板
void DebugPanel::Render(Camera* camera) {
   const ImGuiWindowFlags window_flags = 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.35f); // 半透明背景
    
    if (ImGui::Begin("调试面板", nullptr, window_flags)) {
        // FPS显示
        ImGui::Text("FPS: %d", Time::FPS());
        
        // 摄像机坐标
        const auto& pos = camera->Position;
        ImGui::Text("位置 X: %.1f, Y: %.1f, Z: %.1f)", pos.x, pos.y, pos.z);
        
        // 内存使用
        float memory_usage = Monitor::GetMemoryUsageMB();
        if(memory_usage >= 0) {
            ImGui::Text("内存用量: %.1f MB", memory_usage);
        }
    }
    ImGui::End();
}
}
