// src/ui/panels/pause.cpp
#include "pch.h"
#include "managers/uiMng.h"
#include "ui/panels/pause.h"
#include "ui/settings/content_area.h"

namespace CubeDemo::UI {
// 当前选中的选项卡
static int s_CurrentTab = 0;


void PausePanel::Render(GLFWwindow* window) {
    const ImVec2 window_center = UIMng::GetWindowCenter(window);
    
    // 调整窗口大小
    m_MenuSize = ImVec2(1680, 1024);
    
    // 居中显示
    ImGui::SetNextWindowPos(window_center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(m_MenuSize, ImGuiCond_Always);
    
    // 半透明背景效果
    ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
    
    if (ImGui::BeginPopupModal("PauseMenu", nullptr, 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoScrollbar
    )) 
    {
        // 获取窗口位置和大小
        ImVec2 p_min = ImGui::GetWindowPos();
        ImVec2 p_max = ImVec2(p_min.x + m_MenuSize.x, p_min.y + m_MenuSize.y);
        const float corner_rounding = 12.0f;
        
        // 绘制各个组件
        RenderBackground(p_min, p_max, corner_rounding);
        RenderTitleArea();
        RenderSearchBar();
        RenderContentArea(m_MenuSize);
        
        // 底部区域
        const float buttonAreaTopY = m_MenuSize.y * 0.85f;
        RenderBottomButtons(window, buttonAreaTopY);
        RenderCopyright(buttonAreaTopY);
        
        ImGui::EndPopup();
    } else {
        ImGui::OpenPopup("PauseMenu");
    }
    
    ImGui::PopStyleColor();
}

// 绘制背景
void PausePanel::RenderBackground(ImVec2 p_min, ImVec2 p_max, float corner_rounding) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    // 渐变背景
    draw_list->AddRectFilledMultiColor(
        p_min, p_max,
        ImColor(0.10f, 0.10f, 0.10f, 0.95f),
        ImColor(0.12f, 0.12f, 0.12f, 0.95f),
        ImColor(0.08f, 0.08f, 0.08f, 0.95f),
        ImColor(0.10f, 0.10f, 0.10f, 0.95f)
    );
    
    // 圆角效果
    draw_list->AddRectFilled(p_min, p_max, ImColor(0.15f, 0.15f, 0.15f, 0.95f), corner_rounding);
}

// 绘制标题区域
void PausePanel::RenderTitleArea() {
    ImGui::SetCursorPos(ImVec2(40, 40));
    ImGui::BeginGroup();
    {
        ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "ICON_FA_GAMEPAD");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "游戏已暂停");
    }
    ImGui::EndGroup();
}

// 绘制搜索栏
void PausePanel::RenderSearchBar() {
    ImGui::SetCursorPos(ImVec2(40, 100));
    ImGui::PushItemWidth(300);
    static char searchText[128] = "";
    ImGui::InputTextWithHint("##Search", "搜索设置...", searchText, IM_ARRAYSIZE(searchText));
    ImGui::PopItemWidth();
}

// 绘制内容区域
void PausePanel::RenderContentArea(const ImVec2& menuSize) {
    ImGui::SetCursorPos(ImVec2(40, 140));
    PausePanelBridge bridge = { menuSize, s_CurrentTab, ImGui::GetWindowDrawList() };
    ContentArea::Render(bridge);
}

// 绘制底部按钮
void PausePanel::RenderBottomButtons(GLFWwindow* window, float buttonAreaTopY) {
    // 退出游戏按钮
    ImGui::SetCursorPos(ImVec2(40, buttonAreaTopY));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.05f, 0.05f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.55f, 0.10f, 0.10f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.35f, 0.03f, 0.03f, 1.0f));
    
    if (ImGui::Button("退出到桌面", ImVec2(180, 60))) {
        glfwSetWindowShouldClose(window, true);
    }
    ImGui::PopStyleColor(3);
    
    // 右侧按钮组
    ImGui::SetCursorPos(ImVec2(m_MenuSize.x - 680, buttonAreaTopY));
    ImGui::BeginGroup();
    {
        // 应用按钮
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.35f, 0.75f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.45f, 0.85f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.10f, 0.30f, 0.65f, 1.0f));

        if (ImGui::Button("应用设置", ImVec2(180, 60))) {
            // 应用设置逻辑
        }
        ImGui::PopStyleColor(3);
        
        ImGui::SameLine(0, 20);
        
        // 重置按钮
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.25f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.35f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
        
        if (ImGui::Button("恢复默认", ImVec2(180, 60))) {
            // 重置设置逻辑
        }
        ImGui::PopStyleColor(3);
        ImGui::SameLine(0, 20);
        
        // 回到游戏按钮
        if (ImGui::Button("回到游戏", ImVec2(180, 60))) {
            INPUTS::s_isGamePaused = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::EndGroup();
}

// 绘制版权信息
void PausePanel::RenderCopyright(float buttonAreaTopY) {
    const char* copyright_text = "CubeDemo Engine By 沫兮花落_忧婼子(Kawaii Yora)";
    const float text_width = ImGui::CalcTextSize(copyright_text).x;
    ImGui::SetCursorPos(ImVec2(
        (m_MenuSize.x - text_width) * 0.5f, 
        buttonAreaTopY + 70
    ));
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 0.8f), copyright_text);
}
} // namespace CubeDemo::UI