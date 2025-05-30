// src/main/cleanup.cpp

#include "kits/imgui.h"
#include "main/cleanup.h"

namespace CubeDemo {

// 外部变量声明
extern std::vector<Model*> MODEL_POINTERS;
extern Shader* MODEL_SHADER;
extern Scene* SCENE_INST;
extern ShadowMap* SHADOW_MAP;

// 清理函数
void Cleanup(GLFWwindow* window, Camera* camera) {

    // 确保资源释放顺序
    RL::Shutdown(); // 先关闭资源加载器
    
    // 等待3秒确保资源释放
    TaskQueue::PushTaskSync([]{ 
        glFinish();
    });

    // 摄像机清理
    Camera::Delete(camera);

    // 模型清理
    Model::DeleteAll(MODEL_POINTERS);

    // 模型着色器清理
    Model::DeleteShader(MODEL_SHADER);

    // 阴影着色器清理
    SHADOW_MAP->DeleteShader();

    // 阴影清理
    ShadowMap::DeleteShadow(SHADOW_MAP);

    // ImGui清理
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // GLFW清理
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "程序正常退出" << std::endl;
}
}   // namespace CubeDemo
