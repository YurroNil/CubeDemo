// src/cleanup.cpp
#include <iostream>
#include "cleanup.h"
#include "utils/imguiKits.h"
#include "resources/modelMng.h"
namespace CubeDemo {

void Cleanup(GLFWwindow* window, Camera* camera) {

    // 堆内存创建的对象清理
    ModelMng::Delete("cube");
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

}