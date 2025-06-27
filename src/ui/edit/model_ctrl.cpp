// src/ui/edit/model_ctrl.cpp
#include "pch.h"
#include "ui/edit/model_ctrl.h"
#include "ui/panels/edit.h"
#include "resources/model.h"

// 外部变量声明
namespace CubeDemo {
    extern std::vector<Model*> MODEL_POINTERS;
}

namespace CubeDemo::UI {
void ModelCtrl::Render(Camera* camera) {
    ImGui::Begin("模型控制", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // 创建选项卡栏
    if (ImGui::BeginTabBar("ModelControlTabs", ImGuiTabBarFlags_None)) {
        // 模型监控选项卡
        if (ImGui::BeginTabItem("ICON_FA_EYE" " 模型监控")) {
            ModelMonitorPanel(camera);
            ImGui::EndTabItem();
        }
        
        // 模型编辑选项卡
        if (ImGui::BeginTabItem("ICON_FA_EDIT" " 模型编辑")) {
            ModelEditPanel();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void ModelCtrl::ModelEditPanel() {
    if (ModelCtrl::s_AvailableModels.empty()) {
        ImGui::Text("没有可用模型");
        return;
    }
    
    // 网格布局
    const float cardWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3.0f;
    const float cardHeight = cardWidth * 1.8f;
    
    ImGui::Text("可用模型:");
    ImGui::Spacing();
    
    // 网格容器
    if (ImGui::BeginChild("ModelGrid", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        for (size_t i = 0; i < ModelCtrl::s_AvailableModels.size(); i++) {
            if (i % 3 != 0) ImGui::SameLine();
            
            const auto& modelName = ModelCtrl::s_AvailableModels[i];
            RenderModelCard(modelName, cardWidth, cardHeight);
        }
    }
    ImGui::EndChild();
}
// 模型监控面板
void ModelCtrl::ModelMonitorPanel(Camera* camera) {
    if (MODEL_POINTERS.empty()) {
        ImGui::Text("场景中没有模型");
        ImGui::Text("从模型库中添加模型开始编辑");
        return;
    }

    // 搜索过滤
    static char searchFilter[128] = "";
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.7f);
    ImGui::InputTextWithHint("##Search", "ICON_FA_SEARCH" " 搜索模型...", searchFilter, IM_ARRAYSIZE(searchFilter));
    
    // 模型列表 - 使用树形视图
    if (ImGui::TreeNodeEx("场景中的模型", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        
        for (auto* model : MODEL_POINTERS) {
            // 应用搜索过滤
            if (searchFilter[0] != '\0' && 
                model->GetName().find(searchFilter) == string::npos) {
                continue;
            }
            
            RenderModelCard(model, camera);
        }
        
        ImGui::TreePop();
    }
}
// 渲染模型卡片（编辑面板）
void ModelCtrl::RenderModelCard(const string& modelName, float width, float height) {
    ImGui::BeginGroup();
    // 为每个卡片生成唯一ID（使用名称+索引）
    static int modelIndex = 0; // 静态计数器确保唯一性
    ImGui::PushID(("ModelCard_" + modelName + std::to_string(modelIndex++)).c_str());
    
    // 卡片背景
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.18f, 0.18f, 0.20f, 1.0f));
    ImGui::BeginChild(modelName.c_str(), ImVec2(width, height), true);
    
    // 模型预览图
    ImVec2 previewSize(width - 20, height * 0.7f);
    ImGui::SetCursorPosX((width - previewSize.x) * 0.5f);
    ImGui::Image((ImTextureID)(intptr_t)0, previewSize, ImVec2(0,0), ImVec2(1,1), ImVec4(0.2f, 0.2f, 0.2f, 1.0f), ImVec4(0.8f, 0.8f, 0.8f, 0.5f));
    
    // 模型名称
    ImGui::SetCursorPosX((width - ImGui::CalcTextSize(modelName.c_str()).x) * 0.5f);
    ImGui::Text("%s", modelName.c_str());
    
    // 添加按钮
    if (ImGui::Button(("添加到场景##" + modelName).c_str(), ImVec2(width - 20, 50))) {
        // MODEL_MNG->AddModelToScene(modelName);
    }
    
    ImGui::EndChild();
    ImGui::PopStyleColor();
    
    ImGui::EndGroup();
}
// 渲染模型卡片
void ModelCtrl::RenderModelCard(Model* model, Camera* camera) {
    const string headerID = model->GetName() + "##" + model->m_ID;
    
    // 卡片样式
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 8));
    
    // 卡片背景
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.18f, 0.18f, 0.20f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.22f, 0.22f, 0.24f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.25f, 0.25f, 0.27f, 1.0f));
    
    // 可折叠卡片标题
    if (ImGui::CollapsingHeader(headerID.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        // 模型基本信息
        ImGui::Indent(10.0f);
        ImGui::TextDisabled("ID: %s", model->m_ID.c_str());
        ImGui::TextDisabled("类型: %s", model->m_Type.c_str());
        
        // 变换控件
        RenderTransformControls(model);
        
        // 操作按钮
        ImGui::Spacing();
        RenderModelActions(model, camera);
        
        ImGui::Unindent(10.0f);
    }
    
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
}
// 渲染变换控件
void ModelCtrl::RenderTransformControls(Model* model) {
    const float dragSpeed = 0.1f;
    const float columnWidth = ImGui::GetContentRegionAvail().x * 0.7f;
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // 位置
    ImGui::Text("位置");
    ImGui::PushID(("Position" + model->m_ID).c_str());
    ImGui::SetNextItemWidth(columnWidth);
    if (ImGui::DragFloat3(string("##Pos_" + model->m_ID).c_str(), &model->m_Position.x, dragSpeed, -50.0f, 50.0f, "%.2f")) {
        model->UpdateModelMatrix();
    }
    ImGui::SameLine();
    if (ImGui::Button("ICON_FA_UNDO", ImVec2(120, 50))) {
        model->m_Position = model->m_PosCopy;
        model->UpdateModelMatrix();
    }
    ImGui::PopID();
    
    // 旋转
    ImGui::Text("旋转");
    ImGui::PushID(("Rotation" + model->m_ID).c_str());
    ImGui::SetNextItemWidth(columnWidth);
    if (ImGui::DragFloat(string("##Rot_" + model->m_ID).c_str(), &model->m_Rotation, 1.0f, 0.0f, 360.0f, "%.1f°")) {
        model->UpdateModelMatrix();
    }
    ImGui::SameLine();
    if (ImGui::Button("ICON_FA_UNDO", ImVec2(120, 50))) {
        model->m_Rotation = model->m_RotCopy;
        model->UpdateModelMatrix();
    }
    ImGui::PopID();
    
    // 缩放
    ImGui::Text("缩放");
    ImGui::PushID(("Scale" + model->m_ID).c_str());
    ImGui::SetNextItemWidth(columnWidth);
    if (ImGui::DragFloat3(string("##Scl_" + model->m_ID).c_str(), &model->m_Scale.x, 0.05f, 0.1f, 10.0f, "%.2f")) {
        model->UpdateModelMatrix();
    }
    ImGui::SameLine();
    if (ImGui::Button("ICON_FA_UNDO", ImVec2(120, 50))) {
        model->m_Scale = model->m_ScaleCopy;
        model->UpdateModelMatrix();
    }
    ImGui::PopID();
}
// 渲染模型操作按钮
void ModelCtrl::RenderModelActions(Model* model, Camera* camera) {
    const float buttonWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    
    // 传送模型到相机
    if (ImGui::Button("ICON_FA_ARROW_RIGHT" " 传送模型", ImVec2(buttonWidth, 50))) {
        model->SetPosition(camera->Position - vec3(0.0f, 4.5f, 0.0f));
    }
    
    // 传送相机到模型
    ImGui::SameLine();
    if (ImGui::Button("ICON_FA_ARROW_LEFT" " 传送相机", ImVec2(buttonWidth, 50))) {
        if (camera) camera->TeleportTo(model->m_Position, 4.5f);
    }
    
    // 删除模型
    if (ImGui::Button("ICON_FA_TRASH" " 删除模型", ImVec2(buttonWidth, 50))) {
        // 删除模型逻辑
    }
    
    // 克隆模型
    ImGui::SameLine();
    if (ImGui::Button("ICON_FA_CLONE" " 克隆模型", ImVec2(buttonWidth, 50))) {
        // 克隆模型逻辑
    }
}
}