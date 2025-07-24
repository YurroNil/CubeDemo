// include/ui/main_menu/menubar.cpp
#include "pch.h"
#include "ui/main_menu/preview_panel.h"
#include "ui/main_menu/panel.h"
#include "utils/font_defines.h"
#include "loaders/texture.h"
#include "managers/scene.h"

using MainMenuPanel = CubeDemo::UI::MainMenuPanel;

namespace CubeDemo { extern Managers::SceneMng* SCENE_MNG; }

namespace CubeDemo::UI::MainMenu {

void PreviewPanel::Render() {
    // 右侧预览区域
    ImGui::BeginChild("PreviewPanel",
        ImVec2(
            ImGui::GetWindowWidth() * PREVIEW_WIDTH_RATIO,
            ImGui::GetWindowHeight() * 0.5f
        ), true,
        ImGuiWindowFlags_NoScrollbar
    );
    
    if (m_selectedScene >= 0 && m_selectedScene < m_sceneList.size()) {
        // 使用正确的SceneItem类型
        const auto& scene = m_sceneList[m_selectedScene];
        
        // 预览标题
        ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.00f), ICON_FA_EYE "  场景预览");
        ImGui::Separator();
        ImGui::Spacing();
        
        // 大预览图
        float previewSize = ImGui::GetContentRegionAvail().x * 0.4f;
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - previewSize) * 0.5f);
        
        GLuint texID = scene.preview->ID.load();
        ImTextureID imguiTexID = reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(texID));
        
        // 添加预览图边框
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        draw_list->AddRectFilled(
            ImVec2(p.x - 5, p.y - 5), 
            ImVec2(p.x + previewSize + 5, p.y + previewSize * 0.6f + 5),
            ImColor(0.12f, 0.12f, 0.14f, 1.0f)
        );
        
        // 绘制预览图
        ImGui::Image(imguiTexID, ImVec2(previewSize, previewSize));
        ImGui::Spacing();
        
        // 场景信息
        ImGui::BeginChild("SceneInfo",
            ImVec2(0, 300), ImGuiChildFlags_AlwaysUseWindowPadding,
            ImGuiWindowFlags_NoScrollWithMouse
        );
        
        // 场景名称
        ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "场景名称:");
        ImGui::SameLine();
        ImGui::Text("%s", scene.name.c_str());
        ImGui::Spacing();
        
        // 场景作者
        ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "作者:");
        ImGui::SameLine();
        ImGui::TextWrapped("%s", scene.author.c_str());
        ImGui::Spacing();

        // 场景描述
        ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "描述:");
        ImGui::SameLine();
        ImGui::TextWrapped("%s", scene.description.c_str());
        ImGui::Spacing();
        
        // 场景路径
        ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "文件路径:");
        ImGui::TextWrapped("%s", scene.path.c_str());
        
        ImGui::EndChild();
        
        // 开始场景按钮
        ImGui::Spacing();
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - 200) * 0.5f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(20, 15));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.35f, 0.75f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.45f, 0.85f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.10f, 0.30f, 0.65f, 1.0f));
        
        if (ImGui::Button("加载场景", ImVec2(200, 60))) {
            // 查找场景ID
            string sceneID;
            const auto& allScenes = SCENE_MNG->GetAllScenes();
            
            // 通过资源路径查找场景ID
            for (const auto& [id, scenePtr] : allScenes) {
                if (scenePtr->GetSceneInfo().resourcePath == scene.path) {
                    sceneID = id;
                    break;
                }
            }
            if (!sceneID.empty()) {
                // 切换场景
                SCENE_MNG->SwitchTo(sceneID);
                MainMenuPanel::s_isMainMenuPhase = false;
            } else {
                std::cerr << "未找到匹配的场景ID: " << scene.path << std::endl;
            }
        }
        
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", scene.name.c_str());
        
        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);
    } else {
        // 未选择场景时的提示
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() * 0.4f);
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - 300) * 0.5f);
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 0.7f), "请从左侧选择一个场景");
    }
    ImGui::EndChild();
}
} // namespace CubeDemo::UI::MainMenu
