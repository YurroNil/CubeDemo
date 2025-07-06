// src/ui/main_menu/bottombar.cpp
#include "pch.h"
#include "ui/main_menu/bottombar.h" 
#include "loaders/font.h"
#include "utils/font_defines.h"

namespace CubeDemo::UI::MainMenu {

void Bottombar::Render() {
    // 底部栏
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 60);
    ImGui::BeginChild("BottomBar", ImVec2(
        ImGui::GetWindowWidth(), 40), true,
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoBackground
    );
    
    // 左侧：设置按钮
    ImGui::SetCursorPosX(40);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.25f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.35f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
    
    if (ImGui::Button(ICON_FA_GEAR " 设置", ImVec2(120, 35))) {
        // 打开设置窗口
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    
    // 右侧：版本信息
    ImGui::SameLine(ImGui::GetWindowWidth() - 280);
    ImGui::Text("Cube Demo v1.0.1 | OpenGL %s", glGetString(GL_VERSION));
    
    ImGui::EndChild();
}
}   // namespace CubeDemo::UI::MainMenu
