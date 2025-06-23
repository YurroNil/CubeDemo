// src/ui/panels/pause.cpp
#include "pch.h"
// UI模块
#include "managers/uiMng.h"
#include "ui/panels/pause.h"
// 核心模块
#include "core/inputs.h"

namespace CubeDemo::UI {

// 渲染调试信息面板
void PausePanel::Render(GLFWwindow* window) {
    if (!Inputs::isGamePaused) return; // 如果游戏未暂停，则直接返回

    const ImVec2 pause_menu_size(400, 350); // 暂停菜单的尺寸
    const ImVec2 window_center = UIMng::GetWindowCenter(window); // 获取窗口中心位置

    // 使用ImGui的弹出窗口状态管理
    if (ImGui::BeginPopupModal("PauseMenu", nullptr, 
        ImGuiWindowFlags_NoResize |    // 禁止调整窗口大小
        ImGuiWindowFlags_NoMove |      // 禁止移动窗口
        ImGuiWindowFlags_NoCollapse))   // 禁止折叠窗口
    {
        SetMenuContent(window); // 渲染暂停菜单内容
        ImGui::EndPopup(); // 结束弹出窗口
    } else {
        // 如果弹出窗口未打开，则自动打开
        ImGui::OpenPopup("PauseMenu");
    }

    // 设置窗口属性（只需在首次渲染时设置）
    static bool first_render = true;
    if (first_render) {
        ImGui::SetNextWindowPos(window_center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f)); // 设置窗口位置为窗口中心
        ImGui::SetNextWindowSize(pause_menu_size, ImGuiCond_Always); // 设置窗口大小
        first_render = false;
    }
}
// 渲染暂停菜单内容
void PausePanel::SetMenuContent(GLFWwindow* window) {
    // 居中显示标题
    ImGui::SetCursorPosX((400 - ImGui::CalcTextSize("Game Paused").x) * 0.5f); // 计算标题的水平居中位置
    ImGui::Text("游戏已暂停."); // 显示标题
    ImGui::Separator(); // 添加分隔线

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(20, 15));

    const ImVec4 primaryColor = ImVec4(0.26f, 0.59f, 0.98f, 0.60f);

    // 按钮布局
    const ImVec2 button_size(280, 60); // 按钮的尺寸
    if (ImGui::Button("回到游戏", button_size)) { // 添加"回到游戏"按钮
        Inputs::ResumeTheGame(window); // 恢复游戏
        ImGui::CloseCurrentPopup(); // 关闭弹出窗口
    }

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 0.60f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.25f, 0.80f));

    if (ImGui::Button("退出到桌面", button_size)) { // 添加"退出到桌面"按钮
        glfwSetWindowShouldClose(window, true); // 设置窗口关闭标志
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}
}
