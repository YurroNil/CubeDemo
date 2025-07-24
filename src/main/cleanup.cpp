// src/main/cleanup.cpp
#include "pch.h"
#include "main/cleanup.h"
#include "resources/model.h"
#include "loaders/resource.h"
#include "loaders/model_initer.h"
#include "managers/model.h"
#include "managers/light.h"

namespace CubeDemo {

// 别名与外部变量声明
using RL = Loaders::Resource;
extern std::vector<Model*> MODEL_POINTERS;
extern ShadowMap* SHADOW_MAP;

// 管理器
extern SceneMng* SCENE_MNG; extern LightMng* LIGHT_MNG;
extern ModelMng* MODEL_MNG;

// 清理函数
void Cleanup(GLFWwindow* window, Camera* camera) {
    
    SCENE_MNG->Cleanup();                 // 场景资源
    RL::Shutdown();                       // 源加载器
    TaskQueue::PushTaskSync([]{ glFinish(); }); // 等待3秒确保资源释放
    Renderer::Cleanup();                  // 渲染器
    Camera::Delete(camera);               // 摄像机
    MODEL_MNG->RmvAllModels();            // 所有模型
    SHADOW_MAP->DeleteShader();           // 阴影着色器
    ShadowMap::DeleteShadow(SHADOW_MAP);  // 阴影贴图
    
    // 管理器
    delete SCENE_MNG; SCENE_MNG = nullptr;
    delete LIGHT_MNG; LIGHT_MNG = nullptr;
    delete MODEL_MNG; MODEL_MNG = nullptr;

    // ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "[CLEANNER] 程序正常退出" << std::endl;
}
}   // namespace CubeDemo
