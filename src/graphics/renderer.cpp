// src/graphics/renderer.cpp
#include "pch.h"
#include "graphics/renderer.h"
#include "scenes/dynamic_scene.h"
#include "managers/scene/mng.h"
#include "ui/main_menu/panel.h"
#include "graphics/ray_tracing.h"

// 别名
using MMP = CubeDemo::UI::MainMenuPanel;

namespace CubeDemo {

// 外部变量声明
extern bool RAY_TRACING_ENABLED, RT_DEBUG;
extern Managers::SceneMng* SCENE_MNG;

void Renderer::Init() {
    // 初始化时设置深度测试
    glEnable(GL_DEPTH_TEST);
}

void Renderer::Cleanup() {
    if(!s_RayTracing) {
        delete s_RayTracing; s_RayTracing = nullptr;
    }
    // 清理屏幕四边形
    if (!m_QuadShader) delete m_QuadShader; m_QuadShader = nullptr;
}

void Renderer::RayTracingEnabled(bool enabled) {
    if(enabled) RAY_TRACING_ENABLED = true;
    else RAY_TRACING_ENABLED = false;
}

// 渲染循环首帧
void Renderer::BeginFrame() {
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // ImGui新帧
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 进入渲染循环后，首帧进行一次初始化工作
    if(!RAY_TRACING_ENABLED && !RT_DEBUG) return;
    if (MMP::s_isMainMenuPhase || s_RayTracing) return;
    Create_RT_Inst();
}

void Renderer::Create_RT_Inst(bool is_scene_switching) {
    // 切换场景时先清理
    if (s_RayTracing && (is_scene_switching || !s_isFirstLoad)) s_RayTracing->Cleanup();
    // 创建光追实例
    if (!s_RayTracing) { s_RayTracing = new RayTracing(); }
    else {
        std::cerr << "光追实例创建失败" << std::endl;
        return;
    }

    s_RayTracing->Init();
    if (s_isFirstLoad) s_isFirstLoad = false;
}

// 渲染循环尾帧
void Renderer::EndFrame(GLFWwindow* window) {
    // 渲染ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    // 交换缓冲区
    glfwSwapBuffers(window);
}

// 渲染全屏四边形
void Renderer::RenderFullscreenQuad() {
    static GLuint screenQuadVAO = 0;
    static GLuint screenQuadVBO = 0;
    
    // 首次使用时初始化全屏四边形
    if (screenQuadVAO == 0) {
        float quadVertices[] = {
            // 位置         // 纹理坐标
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };
        
        glGenVertexArrays(1, &screenQuadVAO);
        glGenBuffers(1, &screenQuadVBO);
        glBindVertexArray(screenQuadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, screenQuadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        
        // 位置属性
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        
        // 纹理坐标属性
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        
        glBindVertexArray(0);
    }
    
    if (!m_QuadShader) {
        m_QuadShader = new Shader(
            VSH_POST_PATH + string("screen_quad.glsl"), FSH_POST_PATH + string("screen_quad.glsl")
        );
        if (m_QuadShader == nullptr) {
            std::cerr << "全屏四边形着色器加载失败" << std::endl;
            return;
        }
    }
    
    // 使用着色器
    m_QuadShader->Use();
    
    // 设置纹理
    m_QuadShader->SetInt("screenTexture", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s_RayTracing->GetOutputTexture());
    
    // 设置色调映射参数
    static float exposure = 1.0f;
    static float gamma = 2.2f;
    
    m_QuadShader->SetFloat("exposure", exposure);
    m_QuadShader->SetFloat("gamma", gamma);
    
    // 渲染全屏四边形
    glBindVertexArray(screenQuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    // 解绑
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
}