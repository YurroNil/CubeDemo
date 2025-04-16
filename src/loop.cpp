// src/loop.cpp
#include "init.h"
#include "loop.h"
#include "mainProgramInc.h"
#include "core/time.h"

namespace CubeDemo {
extern std::vector<Model*> MODEL_POINTERS;
extern Shader* MODEL_SHADER;

void MainLoop(WIN, CAM) {
    while (!Window::ShouldClose()) {

        BeginFrame(camera);    // 开始帧
        HandleInput(window);    // 输入管理
        if (!Inputs::isGamePaused) { UpdateModels(); }    // 模型渲染
        HandleWindowSettings(window);    // 窗口输入设置
        RenderScene(window, camera);
        EndFrameHandling(window);
    }
}

// 开始帧
void BeginFrame(CAM) {
    Renderer::BeginFrame();
    Time::Update();
    UIMng::RenderLoop(Window::GetWindow(), *camera);
}

// 输入管理
void HandleInput(WIN) {
    static float lastEscPressTime = 0.0f;
    const float currentTime = glfwGetTime();
    
    if ((currentTime - lastEscPressTime) > Inputs::escCoolDown) {
        lastEscPressTime = currentTime;
        if (!Inputs::isGamePaused) { Inputs::PauseTheGame(window); }
        else if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) { Inputs::ResumeTheGame(window); }
    }
    
    if (!Inputs::isGamePaused) { Inputs::ProcKeyboard(window, Time::DeltaTime()); }
}

// 模型变换(如旋转)
void UpdateModels() {

}

// 输入窗口设置
void HandleWindowSettings(WIN) {
    Window::UpdateWindowSize(window);    // 更新窗口尺寸
    Window::FullscreenTrigger(window);   // 全屏 
}

// 渲染场景
void RenderScene(WIN, CAM) {
    Window::UpdateWindowSize(window);

/* ------应用模型着色器------ */
    MODEL_SHADER->Use();
    // 到摄像机
    MODEL_SHADER->ApplyCamera(*camera, Window::GetAspectRatio());
    // 到模型
    for(auto* thisModel : MODEL_POINTERS) {
        // thisModel->Draw(*MODEL_SHADER);
        if (camera->isSphereVisible(thisModel->bounds.Center, thisModel->bounds.Rad)) {
            thisModel->Draw(*MODEL_SHADER);
        }
    }

}   // RenderScene

void EndFrameHandling(WIN) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}

}