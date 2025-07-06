// src/ui/settings/control.cpp
#include "pch.h"
#include "ui/settings/control.h"

namespace CubeDemo::UI {
void CtrlSettings::Render() {
    
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "按键绑定");
    ImGui::Separator();
    
    // 按键绑定表格
    if (ImGui::BeginTable("KeyBindings", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("操作");
        ImGui::TableSetupColumn("按键");
        ImGui::TableSetupColumn("修改");
        ImGui::TableHeadersRow();
        
        // 示例绑定
        struct KeyBinding {
            const char* action;
            const char* key;
        };
        
        KeyBinding bindings[] = {
            {"向前移动", "W"},
            {"向后移动", "S"},
            {"向左移动", "A"},
            {"向右移动", "D"},
            {"跳跃", "空格"},
            {"下降", "左Shift"},
            {"编辑模式面板", "E"},
            {"预设库面板", "C"},
            {"调试面板", "F3"},
            {"行为", "鼠标左键"},
            {"交互", "鼠标右键"},
            {"聊天", "Enter"},
            {"暂停菜单", "Esc"}
        };
        
        for (int i = 0; i < IM_ARRAYSIZE(bindings); i++) {
            ImGui::TableNextRow();
            
            // 操作列
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", bindings[i].action);
            
            // 按键列
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s", bindings[i].key);
            
            // 修改列
            ImGui::TableSetColumnIndex(2);
            if (ImGui::SmallButton(("修改##" + std::to_string(i)).c_str())) {
                // 按键重绑定逻辑
            }
        }
        
        ImGui::EndTable();
    }
    
    ImGui::Dummy(ImVec2(0, 20));
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "鼠标设置");
    ImGui::Separator();
    
    // Mouse Sensvty(Mouse Senstivity)的简写, 即鼠标灵敏度
    static float mouseSensvty = 1.0f;
    ImGui::Text("鼠标灵敏度");
    ImGui::SliderFloat("##MouseSensvty", &mouseSensvty, 0.1f, 5.0f, "%.2f");
    
    // 鼠标反转
    static bool invertMouse = false;
    ImGui::Text("反转鼠标Y轴");
    ImGui::SameLine();
    ImGui::Checkbox("##InvertMouse", &invertMouse);
    
    // 鼠标平滑
    static bool mouseSmoothing = true;
    ImGui::Text("鼠标平滑");
    ImGui::SameLine();
    ImGui::Checkbox("##MouseSmoothing", &mouseSmoothing);
}
}