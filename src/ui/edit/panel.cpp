// src/ui/edit/panel.cpp
#include "pch.h"
#include "ui/edit/panel.h"
#include "ui/edit/_all.h"
#include "resources/model.h"
#include "managers/model/mng.h"
#include "utils/font_defines.h"

// 外部变量声明
namespace CubeDemo {
    extern std::vector<Model*> MODEL_POINTERS;
    extern SceneMng* SCENE_MNG;
    extern ModelMng* MODEL_MNG;
}

namespace CubeDemo::UI {

// 初始化函数
void EditPanel::Init() {
    // 备份原始ImGui样式
    ImGuiStyle& style = ImGui::GetStyle();
    m_OriginalStyle = style;
    
    // 初始化模型列表
    for(const auto& model : MODEL_POINTERS) {
        CMTP::s_AvailableModels.push_back(model->GetName());
    }

    // 加载图标字体
    // IconFont::Load();
}

// 主渲染入口函数
void EditPanel::Render(Camera* camera) {
    // 如果不是编辑模式，则返回
    if (!INPUTS::s_isEditMode) return;
    
    // 设置现代深色主题
    ApplyModernDarkTheme();
    
    // 创建主窗口
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size);
    
    // 创建一个名为EditPanel的窗口
    ImGui::Begin("EditPanel", nullptr, 
        ImGuiWindowFlags_NoTitleBar |           // 没有标题栏
        ImGuiWindowFlags_NoCollapse |            // 不可折叠
        ImGuiWindowFlags_NoResize |              // 不可调整大小
        ImGuiWindowFlags_NoMove |                // 不可移动
        ImGuiWindowFlags_NoBringToFrontOnFocus | // 焦点时不会前置
        ImGuiWindowFlags_NoNavFocus |            // 导航焦点
        ImGuiWindowFlags_NoSavedSettings |       // 不保存设置
        ImGuiWindowFlags_NoBackground);          // 无背景
    
    // 计算面板尺寸
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = ImGui::GetContentRegionAvail().y;
    const float panel_width = width * 0.25f; // 比例
    
    // 右侧模型管理面板
    ImGui::SetNextWindowPos(ImVec2(width - panel_width - 20, 20));
    ImGui::SetNextWindowSize(ImVec2(panel_width, height - 40));
    
    CtrlPanel(camera);

    ImGui::End();
    ImGui::PopStyleVar(2);
    
    // 恢复原始样式
    ImGui::GetStyle() = m_OriginalStyle;
}

// 应用现代深色主题
void EditPanel::ApplyModernDarkTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // 基础颜色
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.09f, 0.09f, 0.10f, 0.95f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.11f, 0.11f, 0.12f, 0.95f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.11f, 0.11f, 0.12f, 0.95f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.15f, 0.15f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.22f, 0.23f, 1.00f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.26f, 0.28f, 1.00f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.10f, 0.10f, 0.11f, 1.00f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.20f, 0.50f, 0.90f, 1.00f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.26f, 0.28f, 1.00f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.30f, 0.30f, 0.32f, 1.00f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.26f, 0.28f, 1.00f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.30f, 0.32f, 1.00f);
    style.Colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
    style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.26f, 0.26f, 0.28f, 1.00f);
    style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.30f, 0.30f, 0.32f, 1.00f);
    style.Colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.15f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.45f, 0.85f, 1.00f);
    style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.15f, 0.15f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.20f, 0.45f, 0.85f, 1.00f);
    style.Colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
    
    // 圆角设置
    style.WindowRounding = 8.0f;
    style.ChildRounding = 8.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 8.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.TabRounding = 8.0f;
}

void EditPanel::CtrlPanel(Camera* camera) {
    // 创建不可折叠的窗口容器
    ImGui::Begin(" ", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // 创建选项卡栏：ImGui的核心布局组件，用于组织多个功能面板
    if (ImGui::BeginTabBar("ModelMngTabs", ImGuiTabBarFlags_None)) {
        // 场景管理
        if (ImGui::BeginTabItem(ICON_FA_EDIT " 场景管理")) {
            ScenePanel::Render();
            ImGui::EndTabItem();
        }
        
        // 模型列表选项卡
        if (ImGui::BeginTabItem(ICON_FA_EDIT " 模型列表")) {
            CMTP::Render(camera);    // 渲染模型编辑子面板
            ImGui::EndTabItem();
        }

        // 选项卡
        if (ImGui::BeginTabItem(ICON_FA_EYE " 预制体列表")) {
            PresetTable::Render(camera);
            ImGui::EndTabItem();         // 结束当前选项卡
        }

        ImGui::EndTabBar();  // 必须调用以结束选项卡栏
    }
    ImGui::End();  // 结束窗口
}
} // namespace CubeDemo::UI
