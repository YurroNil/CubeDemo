// src/ui/debugPanel.cpp
#include "ui/debugPanel.h"
#include "core/timeMng.h"
#include "core/systemMonitor.h"
#include <imgui.h>
namespace CubeDemo {


void DebugPanel::Render(const Camera& camera) {
    const ImGuiWindowFlags windowFlags = 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.35f); // 半透明背景
    
    if (ImGui::Begin("调试面板", nullptr, windowFlags)) {
        // FPS显示
        ImGui::Text("FPS: %d", TimeMng::FPS());
        
        // 摄像机坐标
        const auto& pos = camera.Position;
        ImGui::Text("位置 X: %.1f, Y: %.1f, Z: %.1f)", pos.x, pos.y, pos.z);
        
        // 内存使用
        float memUsage = SystemMonitor::GetMemoryUsageMB();
        if(memUsage >= 0) {
            ImGui::Text("内存用量: %.1f MB", memUsage);
        }
    }
    ImGui::End();
}


}

