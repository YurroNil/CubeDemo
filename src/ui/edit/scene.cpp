// src/ui/edit/scene.cpp
#include "pch.h"
#include "ui/edit/scene.h"
#include "managers/sceneMng.h"

// 外部变量声明
namespace CubeDemo {
    extern SceneMng* SCENE_MNG;
}

namespace CubeDemo::UI {
void ScenePanel::Render() {
    ImGui::Begin("场景管理", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);
    
    // 场景预览区域
    RenderScenePreview();
    
    // 场景操作按钮
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    const float buttonWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
    
    // 保存场景按钮
    if (ImGui::Button("ICON_FA_SAVE" " 保存场景", ImVec2(buttonWidth, 50))) {
        // 保存场景逻辑
    }
    
    // 清除场景按钮
    ImGui::SameLine();
    if (ImGui::Button("ICON_FA_TRASH" " 清除场景", ImVec2(buttonWidth, 50))) {
        SCENE_MNG->CleanAllScenes();
    }
    
    // 场景切换
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    ImGui::Text("切换场景:");
    const char* sceneNames[] = { "白天(默认)", "夜晚"};
    const int sceneCount = sizeof(sceneNames) / sizeof(sceneNames[0]);
    
    for (int i = 0; i < sceneCount; i++) {
        if (i > 0) ImGui::SameLine();
        
        bool isCurrent = (SCENE_MNG->Current == static_cast<SceneID>(i));
        if (isCurrent) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.59f, 0.98f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        }
        
        if (ImGui::Button(sceneNames[i], ImVec2(0, 50))) {
            SCENE_MNG->SwitchTo(static_cast<SceneID>(i));
        }
        
        if (isCurrent) {
            ImGui::PopStyleColor(2);
        }
    }
    
    ImGui::End();
}

void ScenePanel::RenderScenePreview() {
    // 场景预览标题
    ImGui::Text("场景预览:");
    ImGui::Spacing();
    
    // 预览区域
    const float previewHeight = ImGui::GetContentRegionAvail().y * 0.4f;
    ImVec2 previewSize(ImGui::GetContentRegionAvail().x, previewHeight);
    
    // 创建预览区域
    ImGui::BeginChild("ScenePreview", previewSize, true, ImGuiWindowFlags_NoScrollbar);
    
    // 这里应该是实际的场景预览渲染，暂时用占位符代替
    ImVec2 p_min = ImGui::GetWindowPos();
    ImVec2 p_max = ImVec2(p_min.x + previewSize.x, p_min.y + previewSize.y);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(p_min, p_max, IM_COL32(30, 30, 35, 255));
    
    // 网格线
    const float gridSize = 20.0f;
    for (float x = p_min.x; x <= p_max.x; x += gridSize) {
        draw_list->AddLine(ImVec2(x, p_min.y), ImVec2(x, p_max.y), IM_COL32(50, 50, 55, 255));
    }
    for (float y = p_min.y; y <= p_max.y; y += gridSize) {
        draw_list->AddLine(ImVec2(p_min.x, y), ImVec2(p_max.x, y), IM_COL32(50, 50, 55, 255));
    }
    
    // 场景名称居中显示
    const char* sceneName = SCENE_MNG->GetCurrentScene.Name().c_str();
    ImVec2 textSize = ImGui::CalcTextSize(sceneName);
    ImVec2 textPos = ImVec2((p_min.x + p_max.x - textSize.x) * 0.5f, (p_min.y + p_max.y - textSize.y) * 0.5f);
    draw_list->AddText(textPos, IM_COL32(200, 200, 200, 255), sceneName);
    
    ImGui::EndChild();
}
}   // namespace UI
