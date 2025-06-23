// src/ui/panels/paralist_area.cpp
#include "pch.h"
#include "ui/panels/paralist_area.h"
#include "ui/panels/presetlist_area.h"

namespace CubeDemo::UI {

// 右侧面板区域——预设参数面板
// 右侧参数编辑面板
void ParalistArea::Render() {
    if (PresetlistArea::s_SelectedPreset == -1) {
        // 无选中预设时的提示
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() * 0.4f);
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("请从左侧选择一个预设").x) * 0.5f);
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "请从左侧选择一个预设");
        return;
    }
    
    // 标题区域
    PanelHeader();
    ImGui::Spacing();
    
    // 参数区域（带滚动条的容器）
    ImGui::BeginChild("ParamsArea", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 1.8), true);
    
    // 模型变换参数组
    TransformSection();
    
    // 颜色参数组
    ColorSection();
    
    // 着色器设置组
    ShaderSection();
    
    // 材质参数组
    MaterialSection();
    
    // 高级设置（折叠面板）
    AdvancedSettings();
    
    ImGui::EndChild();
    
    // 底部操作按钮
    ActionButtons();
}

// 模型变换参数组
void ParalistArea::TransformSection() {
    if (ImGui::CollapsingHeader("模型变换", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        
        // 位置参数
        static float position[3] = {0.0f, 0.0f, 0.0f};
        ImGui::Text("位置");
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat3("##Position", position, 0.1f, -100.0f, 100.0f, "%.1f");
        
        ImGui::Spacing();
        
        // 缩放参数
        static float scale[3] = {1.0f, 1.0f, 1.0f};
        ImGui::Text("缩放");
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat3("##Scale", scale, 0.01f, 0.01f, 10.0f, "%.2f");
        
        ImGui::Spacing();
        
        // 旋转参数
        static float rotation[3] = {0.0f, 0.0f, 0.0f};
        static bool alignToCamera = false; // 根据摄像机朝向变换
        
        ImGui::Text("旋转");
        
        // 对齐摄像机复选框
        ImGui::SameLine(ImGui::GetWindowWidth() * 0.7f); // 右侧对齐
        ImGui::Checkbox("根据摄像机朝向变换", &alignToCamera);
        
        // 旋转滑块（根据复选框状态启用/禁用）
        ImGui::BeginDisabled(alignToCamera);
        {
            ImGui::SetNextItemWidth(-1);
            ImGui::DragFloat3("##Rotation", rotation, 1.0f, -180.0f, 180.0f, "%.0f°");
        }
        ImGui::EndDisabled();
        
        // 如果对齐摄像机，显示提示信息
        if (alignToCamera) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 
                "旋转已锁定 - 对象将自动面向摄像机");
        }
        
        ImGui::Separator();
        ImGui::Spacing();
    }
}

// 颜色参数组
void ParalistArea::ColorSection() {
    if (ImGui::CollapsingHeader("颜色与材质", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        
        // 基础颜色选择
        static float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        ImGui::Text("基础颜色");
        ImGui::SetNextItemWidth(-1);
        ImGui::ColorEdit4("##BaseColor", baseColor, 
            ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_DisplayRGB);
        
        ImGui::Spacing();
        
        // 自发光设置
        static float emissionColor[3] = {0.0f, 0.0f, 0.0f};
        static float emissionStrength = 0.0f;
        
        ImGui::Text("自发光");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.4f);
        ImGui::ColorEdit3("##EmissionColor", emissionColor, 
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##EmissionStrength", &emissionStrength, 0.0f, 5.0f, "强度: %.1f");
        
        ImGui::Separator();
        ImGui::Spacing();
    }
}

// 着色器设置组
void ParalistArea::ShaderSection() {
    if (ImGui::CollapsingHeader("着色器设置", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        
        // 着色器预设选择
        static const char* shaderPresets[] = {
            "标准PBR",
            "无光照",
            "卡通着色",
            "线框渲染",
            "自定义"
        };
        static int currentPreset = 0;
        
        ImGui::Text("渲染预设");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##ShaderPreset", &currentPreset, shaderPresets, IM_ARRAYSIZE(shaderPresets))) {
            // 选择预设时更新着色器路径
            if (currentPreset == 4) { // 自定义
                // 允许手动选择着色器
            } else {
                // 应用预设着色器
            }
        }
        
        ImGui::Spacing();
        
        // 顶点着色器选择（仅自定义模式可见）
        if (currentPreset == 4) {
            static const char* vertexShaders[] = {
                "默认顶点着色器",
                "骨骼动画顶点着色器",
                "顶点位移着色器",
                "自定义顶点着色器"
            };
            static int currentVertexShader = 0;
            
            ImGui::Text("顶点着色器");
            ImGui::SetNextItemWidth(-1);
            ImGui::Combo("##VertexShader", &currentVertexShader, vertexShaders, IM_ARRAYSIZE(vertexShaders));
        }
        
        // 片段着色器选择（仅自定义模式可见）
        if (currentPreset == 4) {
            static const char* fragmentShaders[] = {
                "标准PBR片段着色器",
                "无光照片段着色器",
                "卡通渲染片段着色器",
                "水效果片段着色器"
            };
            static int currentFragmentShader = 0;
            
            ImGui::Text("片段着色器");
            ImGui::SetNextItemWidth(-1);
            ImGui::Combo("##FragmentShader", &currentFragmentShader, fragmentShaders, IM_ARRAYSIZE(fragmentShaders));
        }
        
        ImGui::Separator();
        ImGui::Spacing();
    }
}

// 材质参数组
void ParalistArea::MaterialSection() {
    if (ImGui::CollapsingHeader("物理材质")) {
        ImGui::Spacing();
        
        // 金属度
        static float metallic = 0.0f;
        ImGui::Text("金属度");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##Metallic", &metallic, 0.0f, 1.0f, "%.2f");
        
        // 粗糙度
        static float roughness = 0.5f;
        ImGui::Text("粗糙度");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##Roughness", &roughness, 0.0f, 1.0f, "%.2f");
        
        // 法线强度
        static float normalStrength = 1.0f;
        ImGui::Text("法线强度");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##NormalStrength", &normalStrength, 0.0f, 2.0f, "%.1f");
        
        ImGui::Separator();
        ImGui::Spacing();
    }
}

// 高级设置（折叠面板）
void ParalistArea::AdvancedSettings() {
    if (ImGui::CollapsingHeader("高级设置")) {
        ImGui::Spacing();
        
        // 碰撞体设置
        static int collisionType = 0;
        const char* collisionTypes[] = {"无碰撞", "立方体碰撞", "球体碰撞", "网格碰撞"};
        
        ImGui::Text("碰撞体");
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##CollisionType", &collisionType, collisionTypes, IM_ARRAYSIZE(collisionTypes));
        
        // 物理属性
        static float mass = 1.0f;
        static float friction = 0.5f;
        static float bounciness = 0.2f;
        
        ImGui::Text("物理属性");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.3f);
        ImGui::DragFloat("质量", &mass, 0.1f, 0.0f, 100.0f, "%.1f");
        
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.3f);
        ImGui::DragFloat("摩擦", &friction, 0.01f, 0.0f, 1.0f, "%.2f");
        
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("弹性", &bounciness, 0.01f, 0.0f, 1.0f, "%.2f");
        
        // 渲染层设置
        static int renderLayer = 0;
        const char* renderLayers[] = {"默认", "透明", "UI", "特效"};
        
        ImGui::Text("渲染层");
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##RenderLayer", &renderLayer, renderLayers, IM_ARRAYSIZE(renderLayers));
        
        ImGui::Spacing();
    }
}
 
// 面板标题栏（带预设名称显示）
void ParalistArea::PanelHeader() {
    ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "编辑参数"); // 浅灰标题
    ImGui::Spacing();
    
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    ImGui::Text("立方体"); // 动态显示当前选中预设名称
    ImGui::PopStyleColor();
}
// 底部操作按钮组（放置/应用设置）
void ParalistArea::ActionButtons() {
    ImGui::Spacing();
    
    // 主操作按钮样式（蓝色系）
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.59f, 0.98f, 0.80f));      // 默认状态
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.36f, 0.69f, 1.00f, 0.90f));// 悬停状态
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.49f, 0.88f, 1.00f)); // 按下状态
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));            // 文字颜色
    if (ImGui::Button("放置", ImVec2(140, 40))) {
        // 放置逻辑占位（如创建场景物体）
    }
    ImGui::PopStyleColor(4); // 恢复4个样式属性
    
    ImGui::SameLine(); // 按钮水平排列
    
    // 次要操作按钮样式（灰色系）
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.35f, 0.35f, 0.80f));      // 默认状态
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.45f, 0.45f, 0.90f));// 悬停状态
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.25f, 0.25f, 1.00f)); // 按下状态
    if (ImGui::Button("应用设置", ImVec2(200, 40))) {
        PresetlistArea::UpdateSelector(); // 应用参数到选择器
    }
    ImGui::PopStyleColor(3); // 恢复3个样式属性
}
}