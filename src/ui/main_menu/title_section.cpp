// src/ui/main_menu/title_section.cpp
#include "pch.h"
#include "ui/main_menu/title_section.h"
#include "loaders/font.h"

namespace CubeDemo::UI::MainMenu {

void TitleSection::Render() {
    // 使用大字体显示问候语
    ImFont* largeFont = FL::GetLargeTitleFont();
    ImFont* subtitleFont = FL::GetSubtitleFont();
    
    // 顶部大标题区域
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 50);
    
    // 问候语 (晚上好/早上好等)
    if (largeFont) ImGui::PushFont(largeFont, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    
    ImVec2 greetingSize = ImGui::CalcTextSize(m_greeting.c_str());
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - greetingSize.x) * 0.5f);
    ImGui::Text("%s", m_greeting.c_str());
    
    if (largeFont) ImGui::PopFont();
    
    // 副标题 - "欢迎使用CubeDemo"
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 30);
    
    if (subtitleFont) ImGui::PushFont(subtitleFont, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    
    const char* welcomeText = "欢迎使用CubeDemo";
    ImVec2 welcomeSize = ImGui::CalcTextSize(welcomeText);
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - welcomeSize.x) * 0.5f);
    ImGui::Text("%s", welcomeText);
    
    if (subtitleFont) ImGui::PopFont();
    ImGui::PopStyleColor(2);
}
}   // namespace CubeDemo::UI::MainMenu
