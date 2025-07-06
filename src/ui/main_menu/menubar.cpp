// include/ui/main_menu/menubar.h
#include "pch.h"
#include "ui/main_menu/menubar.h"

namespace CubeDemo::UI::MainMenu {

void Menubar::Render() {
    // 增加菜单栏高度
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(15, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(20, 8));
    
    if (ImGui::BeginMainMenuBar()) {
        // 菜单栏高度补偿
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        
        if (ImGui::BeginMenu("文件")) {
            ImGui::MenuItem("新建场景...", nullptr, nullptr, false);
            ImGui::MenuItem("打开场景...", nullptr, nullptr, false);
            ImGui::Separator();
            ImGui::MenuItem("新建存档...", nullptr, nullptr, false);
            ImGui::MenuItem("打开存档...", nullptr, nullptr, false);
            ImGui::Separator();
            if (ImGui::MenuItem("退出", "Alt+F4")) {
                glfwSetWindowShouldClose(WINDOW::GetWindow(), true);
            }
            ImGui::EndMenu();
        }
        
        // 右侧菜单
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 200);
        if (ImGui::BeginMenu("帮助")) {
            if (ImGui::MenuItem("关于CubeDemo")) {
                // 打开关于窗口
            }
            ImGui::EndMenu();
        }
        
        ImGui::EndMainMenuBar();
    }
    
    ImGui::PopStyleVar(2);
}
}   // namespace CubeDemo::UI::MainMenu
