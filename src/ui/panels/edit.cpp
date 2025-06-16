// src/ui/panels/edit.cpp
#include "pch.h"
#include "ui/panels/edit.h"
#include "core/window.h"
#include "resources/model.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
    extern SceneMng* SCENE_MNG;
}

namespace CubeDemo::UI {

std::vector<EditPanel::ModelInfo> EditPanel::s_ModelInfos = {};

std::vector<string> EditPanel::s_AvailableModels = {};

void EditPanel::Render() {
    if (!Inputs::s_isEditMode) return;
    
    // 从场景管理器获取当前场景模型
    s_ModelInfos.clear();
    
    for (const auto& model : MODEL_POINTERS) {
        ModelInfo info;
        info.id = model->GetID();
        info.name = model->GetName();
        info.type = model->GetType();
        info.position = model->GetPosition();
        info.rotation = model->GetRotation();
        info.scale = model->GetScale();
        s_ModelInfos.push_back(info);
    }
    
    // 获取可用模型列表
    if (s_AvailableModels.empty()) s_AvailableModels = SCENE_MNG->GetCurrentScene.ModelNames();

    // 设置主停靠空间（覆盖整个视口）
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::Begin("MainDockspace", nullptr, 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus);
    
    // 创建主停靠节点（右侧区域）
    ImGuiID dockspaceID = ImGui::GetID("EditDockspace");
    ImGui::DockSpace(dockspaceID, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);
    
    // 首次运行时初始化布局
    static bool firstRun = true;
    if (firstRun) {
        firstRun = false;
         // 获取屏幕尺寸
        const float width = ImGui::GetIO().DisplaySize.x;
        const float height = ImGui::GetIO().DisplaySize.y;
        
        // 右上角面板区域（占屏幕25%宽度）
        const float panel_width = width * 0.25f;
        const float panel_height = height / 3;
        
        // 场景面板（顶部）
        ImGui::SetNextWindowPos(ImVec2(width - panel_width, 0));
        ImGui::SetNextWindowSize(ImVec2(panel_width, panel_height));
        SceneMngment();
        
        // 模型监控面板（中部）
        ImGui::SetNextWindowPos(ImVec2(width - panel_width, panel_height));
        ImGui::SetNextWindowSize(ImVec2(panel_width, panel_height));
        ModelMonitoring();
        
        // 模型编辑面板（底部）
        ImGui::SetNextWindowPos(ImVec2(width - panel_width, 2 * panel_height));
        ImGui::SetNextWindowSize(ImVec2(panel_width, panel_height));
        ModelEditing();
    }

    // 独立渲染三个面板（Photoshop风格垂直堆叠）
    ScenePanel(); // 场景管理面板（顶部）
    ModelMonitorPanel(); // 模型监控面板（中部）
    ModelEditPanel(); // 模型编辑面板（底部）

    ImGui::End();
    ImGui::PopStyleVar();
}

// 场景管理面板（右上角顶部）
void EditPanel::ScenePanel() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
    ImGui::Begin("场景管理", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // 标题栏
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "场景控制");
    ImGui::Separator();
    
    // 场景选择按钮组
    if (ImGui::Button("夜晚场景", ImVec2(-1, 30))) SCENE_MNG->SwitchTo(SceneID::NIGHT);
    if (ImGui::Button("默认场景", ImVec2(-1, 30))) SCENE_MNG->SwitchTo(SceneID::DEFAULT);
    
    // 环境参数控制（分组框样式）
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("环境设置");
    static float ambient = 0.35f;
    ImGui::SliderFloat("环境光", &ambient, 0.0f, 1.0f, "%.2f");
    
    ImGui::End();
    ImGui::PopStyleVar();
}

// 模型监控面板（右上角中部）
void EditPanel::ModelMonitorPanel() {
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(5, 10));
    ImGui::Begin("模型监控", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // 遍历所有模型
    for (auto& model : s_ModelInfos) {
        if (ImGui::CollapsingHeader(model.id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Columns(2, "modelProps", false);
            ImGui::SetColumnWidth(0, 100);
            
            // 使用正确的模型引用
            ImGui::Text("位置:"); ImGui::NextColumn();
            ImGui::DragFloat3("##pos", &model.position.x, 0.1f); 
            ImGui::NextColumn();
            
            ImGui::Text("旋转:"); ImGui::NextColumn();
            ImGui::DragFloat("##rot", &model.rotation, 1.0f); 
            ImGui::NextColumn();
            
            ImGui::Columns(1);
        }
    }
    
    ImGui::End();
    ImGui::PopStyleVar();
}

// 模型编辑面板（右上角底部）
void EditPanel::ModelEditPanel() {
    ImGui::Begin("模型编辑", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // 网格布局（2列）
    const float buttonSize = (ImGui::GetWindowWidth() - 20) / 2;
    for (const auto& modelName : s_AvailableModels) {
        if (ImGui::Button(modelName.c_str(), ImVec2(buttonSize, 40))) {
            // 选择逻辑
        }
        // 每两个按钮后换行
        if (ImGui::GetItemRectMax().x < ImGui::GetWindowWidth() - buttonSize) 
            ImGui::SameLine();
    }
    
    ImGui::End();
}


void EditPanel::SceneMngment() {
    ImGui::Text("场景管理");
    ImGui::Separator();
    
    // 获取当前场景
    // auto currentScene = Scenes::SceneMng::GetCurrentSceneName();
    ImGui::Text("当前场景: %s");
    
    // 切换场景按钮
    if (ImGui::Button("上一场景", ImVec2(100, 30))) {
        // Scenes::SceneMng::PreviousScene();
    }
    ImGui::SameLine();
    if (ImGui::Button("下一场景", ImVec2(100, 30))) {
        // Scenes::SceneMng::NextScene();
    }
}

void EditPanel::ModelMonitoring() {
    ImGui::Text("模型监控");
    ImGui::Separator();
    
    for (auto& model : s_ModelInfos) {
        if (ImGui::CollapsingHeader(model.id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            // ID和类型
            ImGui::Text("ID: %s", model.id.c_str());
            ImGui::Text("类型: %s", model.type.c_str());
            
            // 位置编辑
            ImGui::Text("位置:");
            ImGui::PushItemWidth(70);
            ImGui::InputFloat("##posX", &model.position.x, 0.1f, 1.0f, "%.1f");
            ImGui::SameLine();
            ImGui::InputFloat("##posY", &model.position.y, 0.1f, 1.0f, "%.1f");
            ImGui::SameLine();
            ImGui::InputFloat("##posZ", &model.position.z, 0.1f, 1.0f, "%.1f");
            ImGui::SameLine();
            if (ImGui::Button("重置##pos")) {
                model.position = vec3(0.0f);
            }
            ImGui::PopItemWidth();
            
            // 旋转编辑
            ImGui::Text("旋转:");
            ImGui::PushItemWidth(100);
            ImGui::InputFloat("##rotation", &model.rotation, 1.0f, 10.0f, "%.1f°");
            ImGui::SameLine();
            if (ImGui::Button("重置##rot")) {
                model.rotation = 0.0f;
            }
            ImGui::PopItemWidth();
            
            // 缩放编辑
            ImGui::Text("缩放:");
            ImGui::PushItemWidth(70);
            ImGui::InputFloat("##scaleX", &model.scale.x, 0.1f, 1.0f, "%.1f");
            ImGui::SameLine();
            ImGui::InputFloat("##scaleY", &model.scale.y, 0.1f, 1.0f, "%.1f");
            ImGui::SameLine();
            ImGui::InputFloat("##scaleZ", &model.scale.z, 0.1f, 1.0f, "%.1f");
            ImGui::SameLine();
            if (ImGui::Button("重置##scale")) {
                model.scale = vec3(1.0f);
            }
            ImGui::PopItemWidth();
            
            // 操作按钮
            if (ImGui::Button("删除", ImVec2(80, 25))) {
                // 删除逻辑
            }
            ImGui::SameLine();
            if (ImGui::Button("传送到此", ImVec2(100, 25))) {
                // 传送逻辑
            }
            
            ImGui::Separator();
        }
    }
}

void EditPanel::ModelEditing() {
    ImGui::Text("模型编辑");
    ImGui::Separator();
    
    for (const auto& modelName : s_AvailableModels) {
        bool isSelected = (Inputs::s_PlacementModelID == modelName);
        
        if (ImGui::Button(modelName.c_str(), ImVec2(120, 40))) {
            if (isSelected) {
                Inputs::ClearPlacementModel();
            } else {
                Inputs::SetPlacementModel(modelName);
            }
        }
        
        // 高亮显示已选模型
        if (isSelected) {
            ImGui::GetWindowDrawList()->AddRect(
                ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                IM_COL32(0, 255, 0, 255), 3.0f, 0, 2.0f
            );
        }
    }
}
} // namespace CubeDemo::UI