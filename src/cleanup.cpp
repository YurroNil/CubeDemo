// src/cleanup.cpp
#include <iostream>
#include "utils/imguiKits.h"

#include "cleanup.h"
#include "resources/model.h"
#include "core/window.h"
#include "threads/resourceLoader.h"

namespace CubeDemo {
extern std::vector<Model*> MODEL_POINTERS; extern Shader* MODEL_SHADER;

void Cleanup(GLFWwindow* window, Camera* camera) {

    // 确保资源释放顺序
    ResourceLoader::Shutdown(); // 先关闭资源加载器
    
    // 等待3秒确保资源释放
    TaskQueue::PushTaskSync([]{ 
        glFinish();
    });

    // 摄像机清理
    Camera::Delete(camera);
    // 模型清理
    for(auto* thisModel : MODEL_POINTERS) { delete thisModel; thisModel = nullptr; } MODEL_POINTERS.clear();
    // 着色器清理
    delete MODEL_SHADER; MODEL_SHADER = nullptr;

    // ImGui清理
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // GLFW清理
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "程序正常退出" << std::endl;
}

}