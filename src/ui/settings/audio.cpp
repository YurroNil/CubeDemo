// src/ui/settings/audio.cpp
#include "pch.h"
#include "ui/settings/audio.h"

namespace CubeDemo::UI {

void AudioSettings::Render() {
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "音量设置");
    ImGui::Separator();
    
    // 主音量
    static float masterVolume = 0.8f;
    ImGui::Text("主音量");
    ImGui::SliderFloat("##MasterVolume", &masterVolume, 0.0f, 1.0f, "%.2f");
    
    // 音乐音量
    static float musicVolume = 0.7f;
    ImGui::Text("音乐音量");
    ImGui::SliderFloat("##MusicVolume", &musicVolume, 0.0f, 1.0f, "%.2f");
    
    // 音效音量
    static float sfxVolume = 0.9f;
    ImGui::Text("音效音量");
    ImGui::SliderFloat("##SFXVolume", &sfxVolume, 0.0f, 1.0f, "%.2f");
    
    // 环境音量
    static float ambientVolume = 0.6f;
    ImGui::Text("环境音量");
    ImGui::SliderFloat("##AmbientVolume", &ambientVolume, 0.0f, 1.0f, "%.2f");
    
    ImGui::Dummy(ImVec2(0, 20));
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "高级设置");
    ImGui::Separator();
    
    // 音频设备选择
    static int audioDevice = 0;
    ImGui::Text("音频设备");
    if (ImGui::BeginCombo("##AudioDevice", "默认设备")) {
        ImGui::Selectable("默认设备", audioDevice == 0);
        ImGui::Selectable("扬声器 (Realtek)", audioDevice == 1);
        ImGui::Selectable("耳机", audioDevice == 2);
        ImGui::EndCombo();
    }
    
    // 空间音频
    static bool spatialAudio = true;
    ImGui::Text("空间音频");
    ImGui::SameLine();
    ImGui::Checkbox("##SpatialAudio", &spatialAudio);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("很显然这只是个摆设喵~QwQ");
        ImGui::EndTooltip();
    }
}
}   // namespace CubeDemo::UI
