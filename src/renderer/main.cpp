// src/renderer/main.cpp

#include "renderer/main.h"


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

void Renderer::SetLitParameter(Shader& shader, Camera& camera, ModelData& cubeData) {

    shader.SetVec3("light.position", vec3(-0.5f, 2.0f, -2.0f));
    shader.SetVec3("light.color", vec3(1.0f, 1.0f, 1.0f));
    shader.SetFloat("light.ambientStrength", 0.1f);
    shader.SetFloat("light.specularStrength", 0.5f);
    shader.SetFloat("light.constant", 1.0f);
    shader.SetFloat("light.linear", 0.09f);
    shader.SetFloat("light.quadratic", 0.032f);
    
    shader.SetVec3("viewPos", camera.Position);

    // 设置材质属性
    shader.SetVec3("material.ambient", cubeData.material.ambient);
    shader.SetVec3("material.diffuse", cubeData.material.diffuse);
    shader.SetVec3("material.specular", cubeData.material.specular);
    shader.SetFloat("material.shininess", cubeData.material.shininess);

}

void Renderer::Submit(const Shader& shader, const Mesh& mesh, const mat4& model) {
    shader.Use();
    shader.SetMat4("model", model);
    mesh.Draw();
}

void Renderer::ApplyCamera(const Shader& shader, const Camera camera, float aspectRatio) {
    shader.ApplyCamera(camera, aspectRatio);
}

// 渲染循环结束帧
void Renderer::EndFrame(GLFWwindow* window) {

    ImGui::End();   // ImGui结束帧
    // 渲染ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    
    // 交换缓冲区
    glfwSwapBuffers(window);
}
