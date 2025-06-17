// src/graphics/renderer.cpp
#include "pch.h"
#include "graphics/renderer.h"

namespace CubeDemo {

void Renderer::Init() {
    // 初始化时设置深度测试
    glEnable(GL_DEPTH_TEST);
    
}

// 渲染循环开始帧
void Renderer::BeginFrame() {
    // 清空缓冲区代码

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ImGui新帧
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

// 渲染循环结束帧
void Renderer::EndFrame(GLFWwindow* window) {

    // 渲染ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // 交换缓冲区
    glfwSwapBuffers(window);
}

}