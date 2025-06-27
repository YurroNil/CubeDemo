// src/ui/settings/content_area.cpp
#include "pch.h"
#include "ui/settings/_all.h"

namespace CubeDemo::UI {
void ContentArea::Render(PausePanelBridge& bridge) {
    ImGui::BeginChild("ContentArea", ImVec2(
        bridge.MenuSize.x - 80, 
        bridge.MenuSize.y - 340
    ), true);
    
    // 左侧选项卡区域 (25%宽度)
    ImGui::BeginChild("TabBar", ImVec2(ImGui::GetWindowWidth() * 0.25f, 0), true);
    {
        const char* tabNames[] = {"音频", "图像", "控制", "游戏", "关于"};
        const char* tabIcons[] = {"ICON_FA_VOLUME_UP", "ICON_FA_IMAGE", "ICON_FA_KEYBOARD", "ICON_FA_GAMEPAD", "ICON_FA_INFO_CIRCLE"};
        
        // 记录已压栈的样式变量数量
        int pushedStyleVars = 0;
        
        // 增加选项卡间距（1个样式变量）
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 15));
        pushedStyleVars++;
        
        // 设置圆角按钮样式（2个样式变量）
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(15.0f, 10.0f));
        pushedStyleVars += 2;
        
        for (int i = 0; i < IM_ARRAYSIZE(tabNames); i++) {
            bool isSelected = (bridge.CurrentTab == i);
            
            // 记录已压栈的颜色样式数量
            int pushedColors = 0;
            
            // 设置按钮颜色
            if (isSelected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.59f, 0.98f, 0.8f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.59f, 0.98f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.50f, 0.90f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                pushedColors = 4;
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.15f, 0.5f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.20f, 0.20f, 0.7f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.25f, 0.25f, 0.9f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
                pushedColors = 4;
            }
            
            // 使用按钮代替Selectable
            const char* buttonText = (tabIcons[i] + string("  ") + tabNames[i]).c_str();
            if (ImGui::Button(buttonText, ImVec2(-1, 0))) {
                bridge.CurrentTab = i;
            }
            
            // 添加微妙的选中指示器
            if (isSelected) {
                ImVec2 min = ImGui::GetItemRectMin();
                ImVec2 max = ImGui::GetItemRectMax();
                bridge.draw_list->AddRectFilled(
                    ImVec2(min.x, max.y - 3), 
                    ImVec2(max.x, max.y), 
                    ImColor(0.85f, 0.95f, 1.0f, 1.0f), 
                    0, ImDrawFlags_RoundCornersBottom
                );
            }
            
            // 弹出颜色样式
            ImGui::PopStyleColor(pushedColors);
        }
        
        // 修复点：精确弹出所有样式变量
        ImGui::PopStyleVar(pushedStyleVars);
    }
    ImGui::EndChild(); // 结束左侧选项卡
    
    ImGui::SameLine();
    
    // 右侧内容区域 (75%宽度)
    ImGui::BeginChild("Content", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
    {
        ImGui::PushTextWrapPos();
        
        switch (bridge.CurrentTab) {
            case 0: AudioSettings::Render(); break;
            case 1: VideoSettings::Render(); break;
            case 2: CtrlSettings::Render(); break;
            case 3: GameSettings::Render(); break;
            case 4: AboutSection::Render(); break;
        }
        
        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild(); // 结束右侧内容区域
    
    ImGui::EndChild(); // 结束ContentArea
}
}   // namespace CubeDemo::UI
