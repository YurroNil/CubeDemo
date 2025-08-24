// src/ui/presetlib/presetlist_area.cpp
#include "pch.h"
#include "ui/presetlib/presetlist_area.h"

namespace CubeDemo::UI {
// 左侧面板区域——预设列表区域（含搜索/分类/卡片网格）
void PresetlistArea::Render() {
    // 标题区域（浅灰色文字）
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "预设库");
    ImGui::Separator();
    ImGui::Spacing();
    
    // 搜索输入框（暗色背景）
    static char search[32] = "";
    ImGui::SetNextItemWidth(-1); // 宽度自适应
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
    ImGui::InputTextWithHint("##Search", "搜索预设...", search, IM_ARRAYSIZE(search));
    ImGui::PopStyleColor();
    ImGui::Spacing();
    
    // 分类过滤器（下拉选择框）
    const char* categories[] = {"全部", "几何体", "光源", "功能实体"};
    static int current_cat = 0;
    ImGui::Text("分类:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100); // 留出右侧空间
    ImGui::Combo("##Category", &current_cat, categories, IM_ARRAYSIZE(categories));
    ImGui::Spacing();
    
    // 预设卡片网格（核心展示区域）添加容器以确保正确布局
    ImGui::BeginChild("PresetContainer", ImVec2(0, ImGui::GetContentRegionAvail().y - 50), true);
    DrawPresetGrid();
    ImGui::EndChild();

    // 当前选择状态显示
    ImGui::Spacing();
    ImGui::Text("当前选择: ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.7f, 0.3f, 1.0f)); // 金色文字
    ImGui::Text("%s", m_CurrSelector.empty() ? "无" : m_CurrSelector.c_str());
    ImGui::PopStyleColor();
}

// 预设卡片网格绘制（响应式布局）
void PresetlistArea::DrawPresetGrid() {
    // 获取可用宽度并确保合理的列数
    float available_width = ImGui::GetContentRegionAvail().x;
    const int columns = 4;
    
    // 根据可用宽度计算每列的宽度（考虑间距）
    const float padding = 10.0f;
    float column_width = (available_width - padding * (columns - 1)) / columns;
    column_width = std::max(column_width, 100.0f); // 确保最小宽度
    
    // 设置卡片尺寸（根据列宽）
    const float card_width = column_width;
    const float card_height = card_width / 1.5f; // 16:9比例
    
    // 开始网格布局
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(padding, padding));
    
    // 示例预设数据
    const char* preset_names[] = {"立方体", "球体", "三棱锥", "方向光", "聚光", "点光", "面光", "标记实体"};
    const int preset_count = IM_ARRAYSIZE(preset_names);

    // 使用列API实现真正的四列布局
    if (ImGui::BeginTable("PresetGrid", columns, ImGuiTableFlags_SizingFixedFit)) {
        // 计算需要多少行 (ceil(preset_count/columns))
        const int row_count = (preset_count + columns - 1) / columns;
        
        for (int row = 0; row < row_count; row++) {
            ImGui::TableNextRow();
            
            for (int col = 0; col < columns; col++) {
                ImGui::TableSetColumnIndex(col);
                
                const int index = row * columns + col;
                if (index < preset_count) {
                    DrawPresetCard(preset_names[index], index, card_width, card_height);
                } else {
                    // 绘制空单元格占位符
                    ImGui::Dummy(ImVec2(card_width, card_height));
                }
            }
        }
        ImGui::EndTable();
    }
    
    ImGui::PopStyleVar();
}

// 独立函数：绘制单个预设卡片
void PresetlistArea::DrawPresetCard(const char* name, int id, float width, float height) {
    bool isSelected = (s_SelectedPreset == id);
    ImGui::PushID(id); // 唯一ID防止控件冲突
    
    // 获取当前绘制位置
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImVec2 card_size(width, height);
    
    // 不可见按钮（覆盖整个卡片区域）
    if (ImGui::InvisibleButton("##card", card_size)) {
        s_SelectedPreset = id; // 单击选择
        if (ImGui::IsMouseDoubleClicked(0)) {
            UpdateSelector(); // 双击确认选择
        }
    }
    
    // 绘制卡片背景
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (isSelected) {
        // 选中状态
        draw_list->AddRectFilled(p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(0.26f, 0.59f, 0.98f, 0.3f), 8.0f);
        draw_list->AddRect(p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(0.26f, 0.59f, 0.98f, 1.0f), 8.0f, 0, 2.0f);
    } else {
        // 默认状态
        draw_list->AddRectFilled(p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(0.176f, 0.176f, 0.176f, 1.0f), 8.0f);
    }
    
    // 文本区域（动态计算位置）
    ImVec2 text_size = ImGui::CalcTextSize(name);
    float text_pos_x = p.x + (width - text_size.x) * 0.5f;
    float text_pos_y = p.y + height - text_size.y - 10.0f; // 底部留出10px空间
    
    // 绘制卡片名称
    draw_list->AddText(ImVec2(text_pos_x, text_pos_y), ImColor(1.0f, 1.0f, 1.0f, 1.0f), name);
    
    // 图标区域（顶部30%区域）
    float icon_size = std::min(width * 0.4f, height * 0.3f);
    float icon_x = p.x + (width - icon_size) * 0.5f;
    float icon_y = p.y + height * 0.15f;
    
    // 绘制图标（用圆形代替，实际中应替换为纹理）
    draw_list->AddCircleFilled(ImVec2(icon_x + icon_size * 0.5f, icon_y + icon_size * 0.5f), icon_size * 0.5f, ImColor(0.8f, 0.8f, 0.8f, 1.0f));
    
    // 更新光标位置以确保正确的布局
    // ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + height));
    ImGui::Dummy(card_size);
    ImGui::PopID();
}

// 更新选择器逻辑（需与后端数据同步）
void PresetlistArea::UpdateSelector() {
    // 实现逻辑应包括：
    // 将当前预设参数应用到场景
    // 更新m_CurrSelector状态
    // 触发场景刷新等操作
}
}   // namespace CubeDemo::UI
