// src/rendering/renderer.cpp

#include "rendering/renderer.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

void Renderer::Init() {
    // 初始化时设置深度测试
    glEnable(GL_DEPTH_TEST);
}

void Renderer::BeginFrame() {
    glClearColor(0.8f, 0.94f, 0.98f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::Submit(
    const Shader& shader,
    const Mesh& mesh,
    const glm::mat4& model
) {
    shader.Use();
    shader.SetMat4("model", model);
    mesh.Draw();
}

void Renderer::ApplyCamera(const Shader& shader, const Camera camera, float aspectRatio) {
    shader.ApplyCamera(camera, aspectRatio);
}

void Renderer::EndFrame(GLFWwindow* window) {
    glfwSwapBuffers(window);
}
