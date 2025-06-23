// src/ui/panels/presetlist_area.cpp
#include "pch.h"
#include "ui/panels/presetlist_area.h"

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
    
    // 预设卡片网格（核心展示区域）
    DrawPresetGrid();
    
    // 当前选择状态显示
    ImGui::Spacing();
    ImGui::Text("当前选择: ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.7f, 0.3f, 1.0f)); // 金色文字
    ImGui::Text(m_CurrSelector.empty() ? "无" : m_CurrSelector.c_str());
    ImGui::PopStyleColor();
}

// 预设卡片网格绘制（响应式布局）
void PresetlistArea::DrawPresetGrid() {
    const float padding = 10.0f;
    const float card_width = (ImGui::GetColumnWidth() - padding) / 4.0f; // 每列4张卡片
    const float card_height = card_width / 1.5f; // 16:9比例
    
    // 滚动区域容器
    ImGui::BeginChild("PresetGrid", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), true);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(padding, padding));
    
    // 示例预设数据（实际应从数据源加载）
    const char* preset_names[] = {"立方体", "球体", "三棱锥", "方向光", "聚光", "点光", "面光", "标记实体"};
    const int preset_count = IM_ARRAYSIZE(preset_names);

    // const char* preset_icons[] = {...}; 预留图标数组, 将会使用贴图取代（当前未实现）
 
    // 遍历绘制所有预设卡片
    for (int i = 0; i < preset_count; i++) {
        bool isSelected = (s_SelectedPreset == i);
        
        // 卡片容器（用于布局管理）
        ImGui::BeginGroup();
        ImGui::PushID(i); // 唯一ID防止控件冲突
        
        // 获取当前绘制位置
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImVec2 card_size(card_width, card_height);
        
        // 不可见按钮（覆盖整个卡片区域）
        if (ImGui::InvisibleButton("##card", card_size)) {
            s_SelectedPreset = i; // 单击选择
            if (ImGui::IsMouseDoubleClicked(0)) {
                UpdateSelector(); // 双击确认选择
            }
        }
        
        // 绘制卡片背景（在按钮之上）
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        if (isSelected) {
            // 选中状态：半透明蓝色背景+边框
            draw_list->AddRectFilled(p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(0.26f, 0.59f, 0.98f, 0.3f), 8.0f);
            draw_list->AddRect(p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(0.26f, 0.59f, 0.98f, 1.0f), 8.0f, 0, 2.0f);
        } else {
            // 默认状态：深灰色背景
            draw_list->AddRectFilled(p, ImVec2(p.x + card_size.x, p.y + card_size.y), ImColor(0.176f, 0.176f, 0.176f, 1.0f), 8.0f);
        }
        
        // 图标区域（垂直布局：30%顶部间距 + 居中图标）
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y));
        ImGui::Dummy(ImVec2(0, card_size.y * 0.3f)); // 顶部间距

        ImVec2 text_size = ImGui::CalcTextSize(preset_names[i]);

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (card_width - text_size.x) * 0.5f); // 水平居中

        // ImGui::Text("%s", preset_icons[i]); // 实际应替换为图标渲染
        
        // 文本区域（底部对齐 + 居中显示）
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + card_size.y * 0.7f));
        ImGui::SetCursorPosX((card_width - text_size.x) * 0.5f);
        ImGui::Text("%s", preset_names[i]);
        
        // 设置下一个卡片位置
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + card_size.y));
        
        ImGui::PopID();
        ImGui::EndGroup();
        
        // 每行4个卡片后换行（除最后一行外）
        if ((i + 1) % 4 != 0 && i < IM_ARRAYSIZE(preset_names) - 1) {
            ImGui::SameLine(0.0f, padding);
        }
    }
    
    ImGui::PopStyleVar();
    ImGui::EndChild();
}
// 更新选择器逻辑（需与后端数据同步）
void PresetlistArea::UpdateSelector() {
    // 实现逻辑应包括：
    // 1. 将当前预设参数应用到场景
    // 2. 更新m_CurrSelector状态
    // 3. 触发场景刷新等操作
}
}   // namespace CubeDemo::UI
