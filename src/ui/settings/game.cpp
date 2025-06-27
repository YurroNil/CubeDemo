// src/ui/settings/game.cpp
#include "pch.h"
#include "ui/settings/game.h"

namespace CubeDemo::UI {

// 语言选项
static const char* s_LanguageOptions[] = {
    "简体中文", "繁體中文", "English", "日本語", "한국어"
};
static int s_SelectedLanguage = 0; // 默认简体中文
// 主题界面选项
static const char* s_ThemeOptions[] = {
    "深色(默认)", "浅色", "浅色(Skyhelp风格)"
};
static int s_SelectedTheme = 0; // 默认深色主题

void GameSettings::Render() {
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "游戏设置");
    ImGui::Separator();
    
    // 语言设置
    ImGui::Text("语言");
    ImGui::SetNextItemWidth(200);
    ImGui::Combo("##Language", &s_SelectedLanguage, s_LanguageOptions, IM_ARRAYSIZE(s_LanguageOptions));
    
    // 存档目录
    ImGui::Text("存档目录");
    ImGui::TextDisabled("C:\\Users\\Player\\Documents\\CubeDemo\\Saves");
    ImGui::SameLine();
    if (ImGui::SmallButton("浏览...")) {
        // 打开文件夹选择对话框
    }
    
    ImGui::Dummy(ImVec2(0, 20));
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "网络设置");
    ImGui::Separator();
    
    // 主题模式
    static int ThemeMode = 0;
    ImGui::Text("主题界面");

    ImGui::SetNextItemWidth(200);
    ImGui::Combo("##Theme", &s_SelectedTheme, s_ThemeOptions, IM_ARRAYSIZE(s_ThemeOptions));
    
}
}