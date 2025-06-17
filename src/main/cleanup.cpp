// src/main/cleanup.cpp
#include "pch.h"
#include "main/cleanup.h"
#include "resources/model.h"
#include "loaders/resource.h"

#include "managers/modelMng.h"

namespace CubeDemo {

// 别名与外部变量声明
using RL = Loaders::Resource;
using MMC = Managers::ModelCleanner;
extern std::vector<Model*> MODEL_POINTERS;
extern ShadowMap* SHADOW_MAP;

// 管理器
extern SceneMng* SCENE_MNG;
extern LightMng* LIGHT_MNG;
extern ModelMng* MODEL_MNG;

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

    // 先清理模型着色器
    MODEL_MNG->RmvAllShaders();
    // 再清理所有模型
    MODEL_MNG->RmvAllModels();

    // 阴影着色器清理
    SHADOW_MAP->DeleteShader();

    // 阴影清理
    ShadowMap::DeleteShadow(SHADOW_MAP);

    // 管理器清理
    delete SCENE_MNG; SCENE_MNG = nullptr;
    delete LIGHT_MNG; LIGHT_MNG = nullptr;
    delete MODEL_MNG; MODEL_MNG = nullptr;

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
