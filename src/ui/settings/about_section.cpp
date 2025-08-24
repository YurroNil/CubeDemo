// src/ui/settings/about_section.cpp
#include "pch.h"
#include "ui/settings/about_section.h"

namespace CubeDemo::UI {

void AboutSection::Render() {

    // Logo区域
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - 200) * 0.5f);
    // 假设0是占位纹理ID
    ImGui::Image(
        ImTextureRef( (ImTextureID)(intptr_t) 0 ),
        ImVec2(200, 100)
    );
    
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("CubeDemo Engine v1.0.1").x) * 0.5f);
    ImGui::Text("CubeDemo Engine v1.0.1");
    
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("3D游戏引擎").x) * 0.5f);
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "3D游戏引擎");
    
    ImGui::Dummy(ImVec2(0, 20));
    
    // 开发团队
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "关于开发团队");
    ImGui::Separator();
    

    ImGui::BulletText("↓  很显然并没有团队...全由他一个人完成!  ↓");
    ImGui::BulletText("作者: 沫兮花落_忧婼子(Kwaii Yora)");
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("你问为什么是\"他\"? 因为我是男孩子啦~欸？你问为什么名字带女字旁? \n因为我就叫这个名字啦QwQ");
        ImGui::EndTooltip();
    }

    ImGui::Dummy(ImVec2(0, 20));
    
    // 技术特性
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "技术特性");
    ImGui::Separator();

    ImGui::BulletText("风格化、小清新治愈风格");
    ImGui::BulletText("多线程资源加载");
    ImGui::BulletText("跨平台支持(Windows, Linux, macOS)");
    
    ImGui::Dummy(ImVec2(0, 20));
    

    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "关于作者");
    ImGui::Separator();
    ImGui::TextWrapped("CubeDemo Engine及其所有相关内容全部由作者 沫兮花落_忧婼子(KawaiiYora) 一人独立开发，本软件遵循着MIT协议。");
    
    ImGui::Dummy(ImVec2(0, 10));
    ImGui::Text("B站频道: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "https://space.bilibili.com/23009587");
    
    ImGui::Text("GitHub: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "https://github.com/YurroNil");

    ImGui::Text("QQ: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "1424040082");
}
}