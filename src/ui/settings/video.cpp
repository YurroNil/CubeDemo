// src/ui/settings/video.cpp
#include "pch.h"
#include "ui/settings/video.h"


namespace CubeDemo::UI {

// 分辨率选项
static const char* s_ResolutionOptions[] = {
    "1280x720", "1920x1080", "2560x1440", "3840x2160"
};
static int s_SelectedResolution = 1; // 默认1920x1080

// 刷新率选项
static const char* s_RefreshRateOptions[] = {
    "30 Hz", "60 Hz", "120 Hz", "144 Hz", "240 Hz", "垂直同步"
};
static int s_SelectedRefreshRate = 5; // 默认垂直同步

void VideoSettings::Render() {
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "显示设置");
    ImGui::Separator();
    
    // 分辨率
    ImGui::Text("分辨率");
    ImGui::SetNextItemWidth(200);
    ImGui::Combo("##Resolution", &s_SelectedResolution, s_ResolutionOptions, IM_ARRAYSIZE(s_ResolutionOptions));
    
    // 全屏模式
    static bool fullscreen = true;
    ImGui::Text("全屏模式");
    ImGui::SameLine();
    ImGui::Checkbox("##Fullscreen", &fullscreen);
    
    // 刷新率
    ImGui::Text("刷新率");
    ImGui::SetNextItemWidth(200);
    ImGui::Combo("##RefreshRate", &s_SelectedRefreshRate, s_RefreshRateOptions, IM_ARRAYSIZE(s_RefreshRateOptions));
    
    // 垂直同步
    static bool vsync = true;
    ImGui::Text("垂直同步");
    ImGui::SameLine();
    ImGui::Checkbox("##VSync", &vsync);
    
    // HUD大小
    static int hudSize = 2;
    ImGui::Text("HUD大小");
    ImGui::RadioButton("x1", &hudSize, 1); ImGui::SameLine();
    ImGui::RadioButton("x2", &hudSize, 2); ImGui::SameLine();
    ImGui::RadioButton("x3", &hudSize, 3); ImGui::SameLine();
    ImGui::RadioButton("x4", &hudSize, 4);
    
    ImGui::Dummy(ImVec2(0, 20));
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "高级设置");
    ImGui::Separator();
    
    // 异步加载资源
    static bool asyncLoading = true;
    ImGui::Text("异步加载资源");
    ImGui::SameLine();
    ImGui::Checkbox("##AsyncLoading", &asyncLoading);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "下次启动生效");
    
    // 画质预设
    static int qualityPreset = 2;
    ImGui::Text("画质预设");
    ImGui::SetNextItemWidth(200);
    const char* qualityOptions[] = {"极低", "低", "中", "高", "超高"};
    ImGui::Combo("##QualityPreset", &qualityPreset, qualityOptions, IM_ARRAYSIZE(qualityOptions));
    
    // 抗锯齿
    static int aaMode = 2;
    ImGui::Text("抗锯齿");
    ImGui::SetNextItemWidth(200);
    const char* aaOptions[] = {"关闭", "FXAA", "TAA", "SMAA", "MSAA 2x", "MSAA 4x", "MSAA 8x"};
    ImGui::Combo("##AntiAliasing", &aaMode, aaOptions, IM_ARRAYSIZE(aaOptions));
}
}