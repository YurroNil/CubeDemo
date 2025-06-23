// src/ui/panels/edit.cpp
#include "pch.h"
#include "managers/sceneMng.h"
#include "ui/panels/edit.h"
#include "resources/model.h"
#include "managers/modelMng.h"
#include "core/inputs.h"

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
    m_OriginalBgColor = style.Colors[ImGuiCol_WindowBg];
    m_OriginalChildBg = style.Colors[ImGuiCol_ChildBg];
    m_OriginalPopupBg = style.Colors[ImGuiCol_PopupBg];
    m_OriginalFrameBg = style.Colors[ImGuiCol_FrameBg];
    m_OriginalFrameBgHovered = style.Colors[ImGuiCol_FrameBgHovered];
    m_OriginalFrameBgActive = style.Colors[ImGuiCol_FrameBgActive];
    
    // 初始化模型列表
    s_AvailableModels = SCENE_MNG->GetCurrentScene.ModelNames();
    m_LastSceneID = SCENE_MNG->Current;
}

// 渲染函数
// 编辑面板
void EditPanel::Render(Camera* camera) {
    if (!Inputs::s_isEditMode) return;
    
    // 设置透明样式
    ImGuiStyle& style = ImGui::GetStyle();
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.3f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.2f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.2f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.2f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.3f, 0.3f, 0.3f, 0.3f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.4f, 0.4f, 0.4f, 0.4f);
    
    // 检查场景变化
    if (SCENE_MNG->Current != m_LastSceneID) {
        s_AvailableModels = SCENE_MNG->GetCurrentScene.ModelNames();
        m_LastSceneID = SCENE_MNG->Current;
    }
    
    // 创建主窗口
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size);
    ImGui::Begin("EditPanel", nullptr, 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus);
    
    // 计算面板尺寸
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = ImGui::GetContentRegionAvail().y;
    const float panel_width = width * 0.25f;
    const float panel_height = height / 3;
    
    // 场景面板 - 使用深色背景
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.118f, 0.118f, 0.118f, 0.9f)); // #1e1e1e
    ImGui::SetNextWindowPos(ImVec2(width - panel_width, 0));
    ImGui::SetNextWindowSize(ImVec2(panel_width, panel_height));
    ScenePanel();
    ImGui::PopStyleColor();
    
    // 模型控制面板 - 使用深色背景
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.118f, 0.118f, 0.118f, 0.9f)); // #1e1e1e
    ImGui::SetNextWindowPos(ImVec2(width - panel_width, panel_height));
    ImGui::SetNextWindowSize(ImVec2(panel_width, height - panel_height));
    ModelControlPanel(camera);
    ImGui::PopStyleColor();
    
    ImGui::End();
    ImGui::PopStyleVar();
    
    // 恢复原始样式
    style.Colors[ImGuiCol_WindowBg] = m_OriginalBgColor;
    style.Colors[ImGuiCol_ChildBg] = m_OriginalChildBg;
    style.Colors[ImGuiCol_PopupBg] = m_OriginalPopupBg;
    style.Colors[ImGuiCol_FrameBg] = m_OriginalFrameBg;
    style.Colors[ImGuiCol_FrameBgHovered] = m_OriginalFrameBgHovered;
    style.Colors[ImGuiCol_FrameBgActive] = m_OriginalFrameBgActive;
}

// 场景管理面板
void EditPanel::ScenePanel() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(15, 15));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 15));
    
    ImGui::Begin("场景管理", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::Spacing();

    // 按钮样式
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.25f, 0.80f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.35f, 0.90f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.20f, 0.20f, 1.00f));
    
    if (ImGui::Button("白天(默认)", ImVec2(-1, 40))) SCENE_MNG->SwitchTo(SceneID::DEFAULT);
    if (ImGui::Button("夜晚", ImVec2(-1, 40))) SCENE_MNG->SwitchTo(SceneID::NIGHT);
    if (ImGui::Button("清除所有场景", ImVec2(-1, 40))) SCENE_MNG->CleanAllScenes();
    
    ImGui::PopStyleColor(3);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    ImGui::Text("环境设置");
    
    static float ambient = 0.35f;
    ImGui::SliderFloat("环境光", &ambient, 0.0f, 1.0f, "%.2f");
    
    ImGui::End();
    ImGui::PopStyleVar(2);
}

// 模型控制面板
void EditPanel::ModelControlPanel(Camera* camera) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(15, 15));
    ImGui::Begin("模型控制", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::Spacing();
    
    // 创建选项卡栏
    if (ImGui::BeginTabBar("ModelControlTabs", ImGuiTabBarFlags_None)) {
        // 模型监控选项卡
        if (ImGui::BeginTabItem("模型监控")) {
            ModelMonitorPanel(camera);
            ImGui::EndTabItem();
        }
        
        // 模型编辑选项卡
        if (ImGui::BeginTabItem("模型编辑")) {
            ModelEditPanel();
            ImGui::EndTabItem();
        }
        
        ImGui::EndTabBar();
    }
    
    ImGui::End();
    ImGui::PopStyleVar();
}

// 模型监控面板
void EditPanel::ModelMonitorPanel(Camera* camera) {
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 15));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 8));
    
    if (MODEL_POINTERS.empty()) {
        ImGui::PopStyleVar(2); return;
    }

    const float button_width = (ImGui::GetWindowWidth() - 30) / 2;
    const float drag_speed = 0.1f; // 拖拽速度
    
    for (auto* model : MODEL_POINTERS) {
        string headerID = model->GetName() + "##" + model->m_ID;
        string preffix = "##" + model->m_ID;
        
        // 显示模型详细信息 
        if (!ImGui::CollapsingHeader(headerID.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) continue;

        ImGui::Indent(10);
        
        // 卡片式设计
        ImGui::BeginChild(("Card" + preffix).c_str(), ImVec2(0, 0), true, ImGuiWindowFlags_None);
        
        // 显示模型基本信息
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "模型信息");
        ImGui::Text("ID: %s", model->m_ID.c_str());
        ImGui::Text("类型: %s", model->m_Type.c_str());
        ImGui::Text("路径: %s", model->m_Path.c_str());
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // 位置编辑控件 
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "位置");
        ImGui::PushID(("Position" + preffix).c_str());
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.7f);
        if (ImGui::DragFloat3("", &model->m_Position.x, drag_speed, -50.0f, 50.0f, "%.2f")) {
            model->UpdateModelMatrix();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("重置", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0))) {
            model->m_Position = model->m_PosCopy;
            model->UpdateModelMatrix();
        }
        ImGui::PopID();
        
        // 旋转编辑控件 
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "旋转");
        ImGui::PushID(("Rotation" + preffix).c_str());
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.7f);
        if (ImGui::DragFloat("", &model->m_Rotation, 1.0f, 0.0f, 360.0f, "%.1f°")) {
            model->UpdateModelMatrix();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("重置", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0))) {
            model->m_Rotation = model->m_RotCopy;
            model->UpdateModelMatrix();
        }
        ImGui::PopID();
        
        // 缩放编辑控件 
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "缩放");
        ImGui::PushID(("Scale" + preffix).c_str());
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.7f);
        if (ImGui::DragFloat3("", &model->m_Scale.x, 0.05f, 0.1f, 10.0f, "%.2f")) {
            model->UpdateModelMatrix();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("重置", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0))) {
            model->m_Scale = model->m_ScaleCopy;
            model->UpdateModelMatrix();
        }
        ImGui::PopID();
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // 操作按钮
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 10));
        
        if (ImGui::Button(("传送模型至此" + preffix).c_str(), ImVec2(button_width, 40))) {
            model->SetPosition(camera->Position - vec3(0.0f, 4.5f, 0.0f));
        }
        ImGui::SameLine();
        if (ImGui::Button(("传送至此模型" + preffix).c_str(), ImVec2(button_width, 40))) {
            if (camera) camera->TeleportTo(model->m_Position, 4.5f);
        }
        
        ImGui::PopStyleVar();
        ImGui::EndChild();
        ImGui::Unindent(10);
    }
    ImGui::PopStyleVar(2);
}

// 模型编辑面板 - 卡片式设计
void EditPanel::ModelEditPanel() {
    const float card_width = (ImGui::GetContentRegionAvail().x - 30) / 3;
    const float card_height = card_width * 0.8f;
    const float padding = 10.0f;
    
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(padding, padding));
    
    for (size_t i = 0; i < s_AvailableModels.size(); i++) {
        const auto& modelName = s_AvailableModels[i];
        bool isSelected = true; // 这里应该是根据实际状态判断

        // 卡片容器
        ImGui::BeginGroup(); // 开始一个组
        
        ImGui::PushID(static_cast<int>(i));
        
        // 卡片背景 - 根据选择状态改变
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        if (isSelected) {
            draw_list->AddRectFilled(p, ImVec2(p.x + card_width, p.y + card_height), 
                                    ImColor(0.26f, 0.59f, 0.98f, 0.3f), 8.0f);
            draw_list->AddRect(p, ImVec2(p.x + card_width, p.y + card_height), 
                             ImColor(0.26f, 0.59f, 0.98f, 1.0f), 8.0f, 0, 2.0f);
        } else {
            draw_list->AddRectFilled(p, ImVec2(p.x + card_width, p.y + card_height), 
                                    ImColor(0.176f, 0.176f, 0.176f, 1.0f), 8.0f);
        }
        
        // 图标占位符
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (card_width - ImGui::CalcTextSize("图标1").x) * 0.5f);
        ImGui::Text("图标1");
        
        // 文本区域
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (card_width - ImGui::CalcTextSize(modelName.c_str()).x) * 0.5f);
        ImGui::Text("%s", modelName.c_str());
        
        // 检测点击
        if (ImGui::InvisibleButton("##card", ImVec2(card_width, card_height))) {
            if (isSelected) {
                // 处理点击逻辑
            }
        }
        
        ImGui::PopID();
        ImGui::EndGroup(); // 结束组 - 确保每个 BeginGroup() 都有对应的 EndGroup()
        
        // 每行3个卡片
        if ((i + 1) % 3 != 0 && i < s_AvailableModels.size() - 1) {
            ImGui::SameLine(0.0f, padding);
        }
    }
    ImGui::PopStyleVar();
}
} // namespace CubeDemo::UI