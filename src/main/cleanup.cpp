// src/main/cleanup.cpp
#include "pch.h"
#include "main/cleanup.h"
#include "resources/model.h"
#include "loaders/resource.h"

#include "managers/modelMng.h"
#include "managers/lightMng.h"

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

    // 场景资源
    SCENE_MNG->CleanAllScenes();

    // 确保资源释放顺序
    RL::Shutdown(); // 先关闭资源加载器
    
    // 等待3秒确保资源释放
    TaskQueue::PushTaskSync([]{ 
        glFinish();
    });

    // 摄像机
    Camera::Delete(camera);

    // 所有模型相关的资源(包含模型所使用的网格, 着色器和纹理等)
    MODEL_MNG->RmvAllModels();

    // 阴影着色器
    SHADOW_MAP->DeleteShader();

    // 阴影贴图
    ShadowMap::DeleteShadow(SHADOW_MAP);

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
    
    std::cout << "程序正常退出" << std::endl;
}
}   // namespace CubeDemo
