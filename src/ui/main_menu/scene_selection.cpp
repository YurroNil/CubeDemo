// src/ui/main_menu/scene_selection.cpp
#include "pch.h"
#include "ui/main_menu/scene_selection.h"
#include "utils/font_defines.h"
#include "loaders/font.h"
#include "loaders/texture.h"

namespace CubeDemo::UI::MainMenu {

inline void AddRectFilledRounded(ImDrawList* draw_list, const ImVec2& p_min, const ImVec2& p_max, ImU32 col, float rounding, ImDrawFlags flags = 0) {
    if (rounding > 0.0f) {
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_min.y + rounding), rounding, IM_PI, IM_PI * 1.5f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_min.y + rounding), rounding, IM_PI * 1.5f, IM_PI * 2.0f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_max.y - rounding), rounding, 0.0f, IM_PI * 0.5f);
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_max.y - rounding), rounding, IM_PI * 0.5f, IM_PI);
        draw_list->PathFillConvex(col);
    } else {
        draw_list->AddRectFilled(p_min, p_max, col, 0.0f);
    }
}

// 兼容旧版ImGui的圆角边框绘制函数
inline void AddRectRounded(ImDrawList* draw_list, const ImVec2& p_min, const ImVec2& p_max, ImU32 col, float rounding, float thickness, ImDrawFlags flags = 0) {
    if (rounding > 0.0f) {
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_min.y + rounding), rounding, IM_PI, IM_PI * 1.5f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_min.y + rounding), rounding, IM_PI * 1.5f, IM_PI * 2.0f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_max.y - rounding), rounding, 0.0f, IM_PI * 0.5f);
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_max.y - rounding), rounding, IM_PI * 0.5f, IM_PI);
        draw_list->PathStroke(col, true, thickness);
    } else {
        draw_list->AddRect(p_min, p_max, col, 0.0f, 0, thickness);
    }
}

void SceneSelection::Render() {
    // 左侧场景选择区域 - 调整为25%宽度
    ImGui::BeginChild("SceneSelection", 
        ImVec2(
            ImGui::GetWindowWidth() * SELECTION_WIDTH_RATIO,
            ImGui::GetWindowHeight() * 0.5f
        ), true,
        ImGuiWindowFlags_NoScrollbar
    );
    
    ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.00f), ICON_FA_FOLDER_OPEN "  选择场景");
    ImGui::Separator();
    ImGui::Spacing();
    
    // 搜索输入框
    static char search[32] = "";
    ImGui::SetNextItemWidth(-1);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
    ImGui::InputTextWithHint("##Search", "搜索场景...", search, IM_ARRAYSIZE(search));
    ImGui::PopStyleColor();
    ImGui::Spacing();
    
    // 增加表格间距
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(CARD_SPACING_X, CARD_SPACING_Y));
    
    if (ImGui::BeginTable("SceneGrid", 2, ImGuiTableFlags_SizingFixedFit)) {
        for (int i = 0; i < m_sceneList.size(); i++) {
            if (i % 2 == 0) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
            } else {
                ImGui::TableSetColumnIndex(1);
            }
            
            // 增加卡片间距
            ImGui::Dummy(ImVec2(0, CARD_SPACING_Y / 4));
            SceneCard(m_sceneList[i], i, CARD_WIDTH, CARD_HEIGHT);
            ImGui::Dummy(ImVec2(0, CARD_SPACING_Y / 4));
        }
        ImGui::EndTable();
    }
    
    ImGui::PopStyleVar();
    ImGui::EndChild();
}

void SceneSelection::SceneCard(const SceneInfo& scene, int id, float width, float height) {
    bool isSelected = (m_selectedScene == id);
    ImGui::PushID(id);
    
    // 获取当前绘制位置
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImVec2 card_size(width, height);
    
    // 不可见按钮（覆盖整个卡片区域）
    if (ImGui::InvisibleButton("##card", card_size)) {
        m_selectedScene = id;
    }
    
    // 绘制卡片背景
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    // 卡片圆角
    const float rounding = 12.0f;
    
    // 卡片背景颜色
    ImU32 bgColor = isSelected ? 
        ImColor(0.20f, 0.20f, 0.22f, 1.0f) : 
        ImColor(0.15f, 0.15f, 0.17f, 1.0f);
    
    // 使用兼容方法绘制圆角矩形
    AddRectFilledRounded(draw_list, p, ImVec2(p.x + card_size.x, p.y + card_size.y), bgColor, rounding);
    
    // 绘制高亮边框（选中状态）
    if (isSelected) {
        AddRectRounded(draw_list, p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(0.26f, 0.59f, 0.98f, 1.0f), rounding, 2.0f);
    }
    
    // 添加卡片悬停效果
    if (ImGui::IsItemHovered()) {
        AddRectFilledRounded(draw_list, p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(255, 255, 255, 20), rounding);
    }
    
    // 预览图
    GLuint texID = scene.preview->ID.load();
    ImTextureID imguiTexID = reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(texID));
    
    // 计算预览图位置（居中）
    float imageSize = std::min(width * 0.8f, height * 0.6f);
    float imageX = p.x + (width - imageSize) * 0.5f;
    float imageY = p.y + 15.0f;
    
    // 绘制预览图背景
    AddRectFilledRounded(draw_list, 
        ImVec2(imageX - 5, imageY - 5), 
        ImVec2(imageX + imageSize + 5, imageY + imageSize + 5), 
        ImColor(0.10f, 0.10f, 0.12f, 1.0f), 
        6.0f
    );
    
    // 绘制预览图
    draw_list->AddImageRounded(
        imguiTexID, 
        ImVec2(imageX, imageY), 
        ImVec2(imageX + imageSize, imageY + imageSize),
        ImVec2(0, 0), ImVec2(1, 1),
        IM_COL32_WHITE, 
        6.0f
    );
    
    // 场景名称（居中）
    ImGui::PushFont(FL::GetDefaultFont());
    ImVec2 text_size = ImGui::CalcTextSize(scene.name.c_str());
    float text_pos_x = p.x + (width - text_size.x) * 0.5f;
    float text_pos_y = p.y + height - text_size.y - 15.0f;
    
    // 绘制场景名称背景
    AddRectFilledRounded(draw_list, 
        ImVec2(text_pos_x - 10, text_pos_y - 5), 
        ImVec2(text_pos_x + text_size.x + 10, text_pos_y + text_size.y + 5), 
        ImColor(0.12f, 0.12f, 0.14f, 0.9f), 
        8.0f
    );
    
    // 绘制场景名称
    draw_list->AddText(
        ImVec2(text_pos_x, text_pos_y), 
        isSelected ? ImColor(0.9f, 0.9f, 1.0f, 1.0f) : ImColor(1.0f, 1.0f, 1.0f, 1.0f),
        scene.name.c_str()
    );
    ImGui::PopFont();
    
    // 更新光标位置
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + height));
    
    ImGui::PopID();
}
}   // namespace CubeDemo::UI::MainMenu
