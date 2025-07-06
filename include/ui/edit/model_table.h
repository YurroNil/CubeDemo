// src/ui/edit/model_table.h
#pragma once

namespace CubeDemo::UI {

// 模型列表面板类 (CMTP = CubeDemo::UI::ModelTablePanel)
class ModelTablePanel {
public:
    // 核心渲染函数 - 面板入口点
    static void Render(Camera* camera);
    static void Cleanup();

    // 标记FBO是否已初始化
    static bool s_FBO_Inited;
    // 可用模型列表
    static std::vector<string> s_AvailableModels;

private:
    // 状态变量
    static string m_SelectedModel;  // 当前选中的模型名称
    static ImVec2 m_PreviewSize;         // 预览区域尺寸
    
    
    static void RenderModelList();
    static void RenderModelDetail(Camera* camera);
    static void ModelCard(const string& model_name, float width, float height);
    static void RenderModelPreview();
    static void TransformCtrls(Model* model);
    static void ModelActions(Model* model, Camera* camera);

    // 帧缓冲相关
    static unsigned int m_Framebuffer; // 帧缓冲对象
    static unsigned int m_Texture;     // 颜色附件纹理
    static unsigned int m_RBO;         // 渲染缓冲对象（用于深度和模板测试）
    static Shader* m_PreviewShader;    // 预览用的着色器
    
    // 初始化帧缓冲
    static void InitFramebuffer();
};

} // namespace CubeDemo::UI
using CMTP = CubeDemo::UI::ModelTablePanel;