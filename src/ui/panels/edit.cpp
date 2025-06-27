#include "pch.h"
#include "ui/panels/edit.h"
#include "ui/edit/_all.h"
#include "resources/model.h"

#include "managers/modelMng.h"
// #include "utils/icon_font.h" // 添加图标字体支持

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
    ModelCtrl::s_AvailableModels = SCENE_MNG->GetCurrentScene.ModelNames();
    m_LastSceneID = SCENE_MNG->Current;
    
    // 加载图标字体
    // IconFont::Load();
}

// 渲染函数
void EditPanel::Render(Camera* camera) {
    if (!INPUTS::s_isEditMode) return;
    
    // 设置现代深色主题
    ApplyModernDarkTheme();
    
    // 创建主窗口
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size);
    
    ImGui::Begin("EditPanel", nullptr, 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBackground);
    
    // 计算面板尺寸
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = ImGui::GetContentRegionAvail().y;
    const float panel_width = width * 0.25f; // 比例
    
    // 左侧场景面板
    ImGui::SetNextWindowPos(ImVec2(20, 20));
    ImGui::SetNextWindowSize(ImVec2(panel_width, height - 40));
    ScenePanel::Render();
    
    // 右侧模型控制面板
    ImGui::SetNextWindowPos(ImVec2(width - panel_width - 20, 20));
    ImGui::SetNextWindowSize(ImVec2(panel_width, height - 40));
    
    ModelCtrl::Render(camera);
    
    // 底部状态栏
    RenderStatusBar();
    
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

// 渲染状态栏
void EditPanel::RenderStatusBar() {
    const float height = 30.0f;
    ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetMainViewport()->Size.y - height));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetMainViewport()->Size.x, height));
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 5));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.09f, 0.95f));
    
    ImGui::Begin("StatusBar", nullptr, 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings);
    
    // 左侧：场景信息
    ImGui::Text("场景: %s | 模型: %zu", SCENE_MNG->GetCurrentScene.Name().c_str(), MODEL_POINTERS.size());
    
    // 右侧：性能信息
    ImGui::SameLine(ImGui::GetWindowWidth() - 200);
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}
} // namespace CubeDemo::UI
