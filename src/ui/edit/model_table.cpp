// src/ui/edit/model_table.cpp
#include "pch.h"
#include "ui/edit/model_table.h"
#include "ui/edit/preset_table.h"
#include "resources/model.h"
#include "utils/font_defines.h"

namespace CubeDemo {
    extern std::vector<::CubeDemo::Model*> MODEL_POINTERS;
}

namespace CubeDemo::UI {

// 静态成员初始化
std::vector<string> CMTP::s_AvailableModels;
string CMTP::m_SelectedModel = "";
ImVec2 CMTP::m_PreviewSize = ImVec2(0, 0);
unsigned int CMTP::m_Framebuffer = 0;
unsigned int CMTP::m_Texture = 0;
unsigned int CMTP::m_RBO = 0;
Shader* CMTP::m_PreviewShader = nullptr;
bool CMTP::s_FBO_Inited = false;


void CMTP::Render(Camera* camera) {

    // 创建窗口容器 - 模型列表和详情共享的容器
    ImGui::BeginChild("PresetTableContainer", ImVec2(0, 0), true);
    
    // 状态驱动渲染：根据是否有选中的模型决定显示内容
    if (m_SelectedModel.empty()) {

        // 搜索+过滤功能
        static char searchFilter[128] = "";
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.7f);
        ImGui::InputTextWithHint("##Search", ICON_FA_SEARCH " 搜索模型...", searchFilter, IM_ARRAYSIZE(searchFilter));
        
        // 一级界面：模型列表
        RenderModelList(); 

    } else RenderModelDetail(camera); // 二级界面：模型详情

    ImGui::EndChild();
}

// 一级界面：模型列表
void CMTP::RenderModelList() {
    if (s_AvailableModels.empty()) {
        ImGui::Text("没有可用模型");
        return;
    }
    // 计算卡片尺寸 - 考虑间距
    const float availableWidth = ImGui::GetContentRegionAvail().x;
    const float cardWidth = (availableWidth - ImGui::GetStyle().ItemSpacing.x * 2) / 3.0f;
    const float cardHeight = cardWidth * 1.2f;
    
    ImGui::Text("可用模型:");
    ImGui::Spacing();
    
    // 网格布局
    if (ImGui::BeginChild("ModelGrid", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        for (size_t i = 0; i < s_AvailableModels.size(); i++) {
            if (i % 3 != 0) ImGui::SameLine();
            ModelCard(s_AvailableModels[i], cardWidth, cardHeight);
        }
    }
    ImGui::EndChild();
}

// 模型卡片组件
void CMTP::ModelCard(const string& model_name, float width, float height) {
    ImGui::BeginGroup();
    ImGui::PushID(model_name.c_str());
    
    // 卡片样式
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.18f, 0.18f, 0.20f, 1.0f));

    // 使用固定大小的子区域
    ImGui::BeginChild(model_name.c_str(), ImVec2(width, height), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    
    // 整个卡片作为可点击按钮
    if (ImGui::InvisibleButton("##CardBtn", ImVec2(width, height))) {
        m_SelectedModel = model_name; // 设置选中模型
    }
    
    // 悬停效果 - 直接使用当前项的位置
    const bool isHovered = ImGui::IsItemHovered();
    if (isHovered) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 min = ImGui::GetItemRectMin();
        ImVec2 max = ImGui::GetItemRectMax();
        draw_list->AddRect(min, max, ImGui::GetColorU32(ImVec4(0.26f, 0.59f, 0.98f, 1.0f)), 8.0f, 0, 2.0f);
    }
    
    // 模型预览图 - 使用卡片内部尺寸
    const float previewHeight = height * 0.6f;
    const float previewWidth = width - 20;
    ImVec2 previewPos((width - previewWidth) * 0.5f, 10);
    
    // 设置预览图位置
    ImGui::SetCursorPos(previewPos);
    
    // 使用悬停状态改变色调
    ImVec4 tintColor = isHovered ? 
        ImVec4(1.0f, 1.0f, 1.0f, 1.0f) : 
        ImVec4(0.8f, 0.8f, 0.8f, 1.0f);

    ImGui::Image(
        (ImTextureID)(intptr_t)0, 
        ImVec2(previewWidth, previewHeight), 
        ImVec2(0,0), ImVec2(1,1), 
        tintColor, ImVec4(1,1,1,1)
    ); 
    
    // 模型名称 - 放在预览图下方
    ImGui::SetCursorPos(ImVec2(
        (width - ImGui::CalcTextSize(model_name.c_str()).x) * 0.5f,
        previewHeight + 20
    ));
    ImGui::Text("%s", model_name.c_str());
    
    // 提示文本 - 放在名称下方
    const char* hint = "点击查看详情";
    ImGui::SetCursorPos(ImVec2(
        (width - ImGui::CalcTextSize(hint).x) * 0.5f,
        previewHeight + 40
    ));
    ImGui::TextDisabled("%s", hint);
    
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopID();
    ImGui::EndGroup();
}

// 二级界面：模型详情
void CMTP::RenderModelDetail(Camera* camera) {
    // 获取容器尺寸用于布局
    const ImVec2 containerSize = ImGui::GetContentRegionAvail();
    const float previewHeight = containerSize.y * 0.4f; // 调整为40%
    
    // 顶部返回按钮
    if (ImGui::Button(ICON_FA_ARROW_LEFT " 返回列表", ImVec2(180, 50))) {
        m_SelectedModel.clear(); // 清除选中状态
    }
    
    // 模型名称标题
    ImGui::SameLine();
    float textWidth = ImGui::CalcTextSize(m_SelectedModel.c_str()).x;
    ImGui::SetCursorPosX((containerSize.x - textWidth) * 0.5f);
    ImGui::Text("%s", m_SelectedModel.c_str());
    
    ImGui::Spacing(); // 减少间距
    ImGui::Separator();
    ImGui::Spacing(); // 减少间距
    
    // 模型预览区域 - 使用固定高度
    ImGui::BeginChild("ModelPreview", ImVec2(containerSize.x, previewHeight), true);
    RenderModelPreview();
    ImGui::EndChild();
    
    ImGui::Spacing(); // 减少间距
    
    // 模型控制区域 
    // 查找对应的模型实例
    Model* targetModel = nullptr;
    for (auto* model : MODEL_POINTERS) {
        if (model->GetName() == m_SelectedModel) {
            targetModel = model;
            break;
        }
    }
    // 渲染变换控件和操作按钮
    if (targetModel) {
        TransformCtrls(targetModel);
        ImGui::Spacing();
        ModelActions(targetModel, camera);
    } else {
        ImGui::Text("模型未加载: %s", m_SelectedModel.c_str());
    }
}

// 模型预览渲染
void CMTP::RenderModelPreview() {
    using namespace glm;

    // 确保FBO已初始化
    if (!s_FBO_Inited) InitFramebuffer();
    
    // 保存当前OpenGL状态
    GLint last_framebuffer;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &last_framebuffer);
    GLint last_viewport[4];
    glGetIntegerv(GL_VIEWPORT, last_viewport);
    GLboolean last_depth_test = glIsEnabled(GL_DEPTH_TEST);
    
    // 获取预览区域尺寸
    const ImVec2 preview_size = ImGui::GetContentRegionAvail();
    m_PreviewSize = preview_size; // 存储尺寸供外部使用
    
    // 调整FBO纹理大小以匹配预览窗口
    if (preview_size.x > 0 && preview_size.y > 0) {
        glBindTexture(GL_TEXTURE_2D, m_Texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, preview_size.x, preview_size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        glBindRenderbuffer(GL_RENDERBUFFER, m_RBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, preview_size.x, preview_size.y);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
}
    
    // 绑定FBO并渲染场景
    glBindFramebuffer(GL_FRAMEBUFFER, m_Framebuffer);
    glViewport(0, 0, preview_size.x, preview_size.y);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    
    // 设置透视投影矩阵 - 为预览窗口专门设置
    mat4 projection = perspective(
        radians(45.0f), 
        preview_size.x / preview_size.y, 
        0.1f, 100.0f
    );
    
    // 设置视图矩阵（相机位置）- 为预览窗口专门设置
    mat4 view = lookAt(
        vec3(0.0f, 1.0f, 3.0f),  // 相机位置
        vec3(0.0f, 0.0f, 0.0f),  // 目标位置
        vec3(0.0f, 1.0f, 0.0f)   // 相机朝向
    );
    
    // 使用预览着色器
    m_PreviewShader->Use();
    m_PreviewShader->SetMat4("projection", projection);
    m_PreviewShader->SetMat4("view", view);
    
    // 查找选中的模型并渲染
    if (!m_SelectedModel.empty()) {
        Model* model = nullptr;
        for (auto m : MODEL_POINTERS) {
            if (m->GetName() == m_SelectedModel) {
                model = m;
                break;
            }
        }
        if (model) {
            mat4 modelMat = mat4(1.0f);

            // 为了预览，我们可以将模型放在原点，并应用一个固定的缩放
            // 只有预览模型的位置是锁在(0, -1, 0)的，其他参数使用源模型的属性
            modelMat = translate(modelMat, vec3(0.0f, -1.0f, 0.0f));

            // 因为模型的缩放是相对的，所以我们需要将其缩放到适合预览窗口的尺寸
            modelMat = scale(modelMat, model->m_Scale * 0.3f);

            modelMat = rotate(modelMat, model->m_Rotation.x, vec3(1.0f, 0.0f, 0.0f));
            modelMat = rotate(modelMat, model->m_Rotation.y, vec3(0.0f, 1.0f, 0.0f));
            modelMat = rotate(modelMat, model->m_Rotation.z, vec3(0.0f, 0.0f, 1.0f));
            
            m_PreviewShader->SetMat4("model", modelMat);
            
            // 设置纹理位置（预览着色器只需要一个基础纹理）
            m_PreviewShader->SetInt("texture_diffuse1", 0);
            
            model->DrawCall(nullptr, false);
        }
    }
    
    // 解绑FBO
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // 恢复OpenGL状态
    glBindFramebuffer(GL_FRAMEBUFFER, last_framebuffer);
    glViewport(last_viewport[0], last_viewport[1], last_viewport[2], last_viewport[3]);
    
    if (last_depth_test) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    
    // 在ImGui中显示纹理
    ImGui::Image((void*)(intptr_t)m_Texture, preview_size, ImVec2(0,1), ImVec2(1,0)); // 注意纹理坐标翻转
    
    // 添加叠加信息（可选）
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 p_min = ImGui::GetCursorScreenPos();
    ImVec2 p_max = ImVec2(p_min.x + preview_size.x, p_min.y + preview_size.y);
    
    // 提示文本
    const char* hint = "模型预览区域";
    ImVec2 hintSize = ImGui::CalcTextSize(hint);
    ImVec2 hintPos(p_min.x + 10, p_max.y - hintSize.y - 10);
    draw_list->AddText(hintPos, IM_COL32(150, 150, 150, 180), hint);
}

// 初始化帧缓冲
void CMTP::InitFramebuffer() {
    if (s_FBO_Inited) return;
    
    // 创建帧缓冲
    glGenFramebuffers(1, &m_Framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_Framebuffer);
    
    // 创建纹理附件
    glGenTextures(1, &m_Texture);
    glBindTexture(GL_TEXTURE_2D, m_Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_Texture, 0);
    
    // 创建渲染缓冲对象（用于深度和模板附件）
    glGenRenderbuffers(1, &m_RBO);
    glBindRenderbuffer(GL_RENDERBUFFER, m_RBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 800, 600);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_RBO);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer is not complete!" << std::endl;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // 创建简化的预览着色器
    m_PreviewShader = new Shader(
        VSH_PATH + string("preview.glsl"),
        FSH_PATH + string("preview.glsl")
    );
    
    s_FBO_Inited = true;
}

// 模型变换控件组
void CMTP::TransformCtrls(Model* model) {
    const float columnWidth = ImGui::GetContentRegionAvail().x * 0.7f;
    
    ImGui::Spacing();
    ImGui::Separator();  // 视觉分隔线
    ImGui::Spacing();
    
    // 模型基本信息显示
    ImGui::TextDisabled("ID: %s", model->m_ID.c_str());
    ImGui::TextDisabled("类型: %s", model->m_Type.c_str());

    // 位置控件
    ImGui::Text("位置");
    ImGui::PushID(("Position" + model->m_ID).c_str());
    ImGui::SetNextItemWidth(columnWidth);
    // 三轴拖拽控件
    if (ImGui::DragFloat3(string(
        "##Pos_" + model->m_ID).c_str(),
        &model->m_Position.x,
        0.05f, -50.0f, 50.0f, "%.2f"
    )) { model->UpdateModelMatrix(); }
    
    ImGui::SameLine();  // 同行显示重置按钮
    if (ImGui::Button(ICON_FA_UNDO, ImVec2(120, 50))) {
        model->m_Position = model->m_PosCopy;
        model->UpdateModelMatrix();
    }
    ImGui::PopID();
    
    // 旋转控件
    ImGui::Text("旋转");
    ImGui::PushID(("Rotation" + model->m_ID).c_str());
    ImGui::SetNextItemWidth(columnWidth);

    if (ImGui::DragFloat3(string(
        "##Rot_" + model->m_ID).c_str(), 
        &model->m_Rotation.x,
        0.01f, 0.0f, 6.28f, "%.2f"
        )) { model->UpdateModelMatrix(); }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_UNDO, ImVec2(120, 50))) {
        model->m_Rotation = model->m_RotCopy;
        model->UpdateModelMatrix();
    }
    ImGui::PopID();
    
    // 缩放控件
    ImGui::Text("缩放");
    ImGui::PushID(("Scale" + model->m_ID).c_str());
    ImGui::SetNextItemWidth(columnWidth);
    if (ImGui::DragFloat3(string(
        "##Scl_" + model->m_ID).c_str(), 
        &model->m_Scale.x,
        0.01f, 0.1f, 10.0f, "%.2f"
        )) { model->UpdateModelMatrix(); }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_UNDO, ImVec2(120, 50))) {
        model->m_Scale = model->m_ScaleCopy;
        model->UpdateModelMatrix();
    }
    ImGui::PopID();
}

// 模型操作按钮组
void CMTP::ModelActions(Model* model, Camera* camera) {

    const float buttonWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    
    // 传送模型到相机位置
    ImGui::PushID(("##TPModel_toCam_"+ model->m_ID).c_str());
    if (ImGui::Button(ICON_FA_ARROW_RIGHT " 传送模型此", ImVec2(buttonWidth, 50))) {
        if(camera != nullptr) model->SetPosition(camera->Position - vec3(0.0f, 4.5f, 0.0f));
    }
    ImGui::PopID();

    // 传送相机到模型位置
    ImGui::SameLine();
    ImGui::PushID(("##TPtoModel_"+ model->m_ID).c_str());
    if (ImGui::Button(ICON_FA_ARROW_LEFT " 传送相机至模型", ImVec2(buttonWidth, 50))) {
        if (camera != nullptr) camera->TeleportTo(model->m_Position, 4.5f);  // 引擎集成点
    }
    ImGui::PopID();

    // 删除按钮
    ImGui::PushID(("##Remove_"+ model->m_ID).c_str());
    if (ImGui::Button(ICON_FA_TRASH " 删除模型", ImVec2(buttonWidth, 50))) {
        model->Delete();
    }
    ImGui::PopID();

    // 克隆按钮
    ImGui::SameLine();
    ImGui::PushID(("##Clone_"+ model->m_ID).c_str());
    if (ImGui::Button(ICON_FA_CLONE " 克隆模型", ImVec2(buttonWidth, 50))) {
        // 克隆模型逻辑
    }
    ImGui::PopID();
}

void CMTP::Cleanup() {
    if (s_FBO_Inited) {
        glDeleteFramebuffers(1, &m_Framebuffer);
        glDeleteTextures(1, &m_Texture);
        glDeleteRenderbuffers(1, &m_RBO);
        delete m_PreviewShader;
        s_FBO_Inited = false;
    }
}
} // namespace CubeDemo::UI
