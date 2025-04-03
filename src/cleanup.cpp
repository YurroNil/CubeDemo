// src/cleanup.cpp

#include "cleanup.h"
#include "resources/modelManager.h"


void CubeDemo::Cleanup(GLFWwindow* window, Camera* camera) {

    // 堆内存创建的对象清理
    ModelManager::Delete("cube");
    Camera::Delete(camera);

    // ImGui清理
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // GLFW清理
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "程序正常退出" << std::endl;
}
