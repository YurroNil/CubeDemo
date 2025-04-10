// src/graphics/mainRenderer.cpp

#include "graphics/mainRenderer.h"
#include "utils/imguiKits.h"

namespace CubeDemo {

void Renderer::Init() {
    // 初始化时设置深度测试
    glEnable(GL_DEPTH_TEST);
}


// 渲染循环开始帧
void Renderer::BeginFrame() {
    // 清空缓冲区代码
    glClearColor(0.8f, 0.94f, 0.98f, 1.0f);
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