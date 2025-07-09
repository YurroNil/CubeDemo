// src/ui/edit/scene.cpp
#include "pch.h"
#include "ui/edit/scene.h"
#include "utils/font_defines.h"
#include "loaders/model_initer.h"

// 外部变量声明
namespace CubeDemo { extern SceneMng* SCENE_MNG; }

namespace CubeDemo::UI {

void ScenePanel::SaveCurrentScene() {
    auto* scene = SCENE_MNG->GetCurrentScene();
    if (!scene) return;
    
    // TODO: 实现场景保存逻辑
    // 收集当前场景的所有修改
    // 更新场景配置
    // 保存到对应的scene_info.json文件
    
    // 示例：获取场景ID
    string sceneID = scene->GetID();
    std::cout << "保存场景: " << sceneID << std::endl;
}

void ScenePanel::ClearCurrentScene() {
    auto* scene = SCENE_MNG->GetCurrentScene();
    if (!scene) return;
    
    // 清除场景资源
    scene->Cleanup();
    
    // 可以添加确认对话框
    ImGui::OpenPopup("确认清除场景");
}

void ScenePanel::Render() {
    ImGui::BeginChild("场景管理", ImVec2(0, 0), true);
    
    // 场景预览区域
    ScenePreview();
    
    // 场景操作按钮
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    const float buttonWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
    
    // 保存场景按钮
    if (ImGui::Button(ICON_FA_SAVE " 保存场景", ImVec2(buttonWidth, 50))) {
        SaveCurrentScene();
    }
    
    // 清除场景按钮
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH " 清除场景", ImVec2(buttonWidth, 50))) {
        ClearCurrentScene();
    }
    
    // 场景切换
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    ImGui::Text("切换场景:");
    
    // 获取所有可用场景
    const auto& allScenes = SCENE_MNG->GetAllScenes();
    auto* currentScene = SCENE_MNG->GetCurrentScene();
    string currentSceneID = currentScene ? currentScene->GetID() : "";
    
    // 为每个场景创建切换按钮
    for (const auto& [sceneID, scene] : allScenes) {
        bool isCurrent = (sceneID == currentSceneID);
        
        if (isCurrent) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.59f, 0.98f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        }
        
        // 使用场景名称作为按钮标签
         if (ImGui::Button(scene->GetName().c_str(), ImVec2(0, 50))) {
            try {
                SCENE_MNG->SwitchTo(sceneID);
            } catch (const std::exception& e) {
                std::cerr << "场景切换失败: " << e.what() << std::endl;
            }
        }
        
        if (isCurrent) ImGui::PopStyleColor(2);
        
        // 添加工具提示显示场景ID
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("ID: %s", sceneID.c_str());
            ImGui::EndTooltip();
        }
        
        // 在同一行显示，但每行最多显示2个按钮
        if (&scene != &allScenes.begin()->second) {
            ImGui::SameLine();
        }
    }
    
    ImGui::EndChild();
}

void ScenePanel::ScenePreview() {
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
    const char* sceneName = SCENE_MNG->GetCurrentScene()->GetName().c_str();
    ImVec2 textSize = ImGui::CalcTextSize(sceneName);
    ImVec2 textPos = ImVec2((p_min.x + p_max.x - textSize.x) * 0.5f, (p_min.y + p_max.y - textSize.y) * 0.5f);
    draw_list->AddText(textPos, IM_COL32(200, 200, 200, 255), sceneName);
    
    ImGui::EndChild();
}
}   // namespace UI
