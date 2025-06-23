// src/ui/panels/presetlib.cpp
#include "pch.h"
#include "ui/panels/presetlib.h"
#include "ui/panels/presetlist_area.h"
#include "ui/panels/paralist_area.h"

namespace CubeDemo::UI {

// 预设库面板初始化（单例模式）
void PresetlibPanel::Init() {
    s_initialized = true; // 标记面板已初始化
}
 
// 预设库主渲染函数（核心UI逻辑）
void PresetlibPanel::Render(Camera* camera) {
    // 窗口初始化设置（仅首次显示时生效）
    ImGui::SetNextWindowSize(ImVec2(1280, 1050), ImGuiCond_FirstUseEver);
    
    // 样式预设：窗口内边距15px，控件间距15x12，输入框内边距12x8
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(15, 15));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(15, 12));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
    
    // 开始窗口（带菜单栏和固定大小）
    if (!ImGui::Begin("预设库", nullptr, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoResize)) {
        ImGui::End();
        ImGui::PopStyleVar(3); // 恢复之前保存的样式
        return;
    }
    
    // 绘制顶部菜单栏
    DrawMenuBar();
    
    // 布局系统：分割为2列.
    // 第1列: 左侧面板区域——预设列表(占35%宽度);
    // 第2列: 右侧面板区域——预设参数面板(宽度自适应);
    ImGui::Columns(2, "PresetColumns", true);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.35f);
    
    // 预设列表样式（深色背景 #1e1e1e）
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.118f, 0.118f, 0.118f, 1.0f));

    // 预设列表
    PresetlistArea::Render();

    ImGui::PopStyleColor();
    ImGui::NextColumn(); // 切换到右侧列
    
    // 预设参数面板样式（稍亮背景 #252525）
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.145f, 0.145f, 0.145f, 1.0f));

    // 预设参数面板
    ParalistArea::Render();
    ImGui::PopStyleColor();
    
    // 恢复单列布局
    ImGui::Columns(1);
    ImGui::End();
    ImGui::PopStyleVar(3); // 恢复之前保存的样式
}
 
// 顶部菜单栏绘制（文件/编辑/视图菜单）
void PresetlibPanel::DrawMenuBar() {
    if (!ImGui::BeginMenuBar()) return;
    
    // 文件菜单（带快捷键提示）
    if (ImGui::BeginMenu("文件")) {
        if (ImGui::MenuItem("保存预设", "Ctrl+S")) {}      // 保存功能占位
        if (ImGui::MenuItem("加载预设", "Ctrl+O")) {}      // 加载功能占位
        ImGui::EndMenu();
    }
    
    // 编辑菜单
    if (ImGui::BeginMenu("编辑")) {
        if (ImGui::MenuItem("新建预设", "Ctrl+N")) {}      // 新建功能占位
        if (ImGui::MenuItem("复制预设", "Ctrl+C")) {}      // 复制功能占位
        if (ImGui::MenuItem("删除预设", "Del")) {}         // 删除功能占位
        ImGui::EndMenu();
    }
    
    // 视图菜单（切换显示模式）
    if (ImGui::BeginMenu("视图")) {
        ImGui::MenuItem("大图标", nullptr, "预览大图标");   // 视图模式切换占位
        ImGui::MenuItem("显示描述", nullptr, "预览描述");   // 描述显示切换占位
        ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
}
} // namespace CubeDemo::UI
