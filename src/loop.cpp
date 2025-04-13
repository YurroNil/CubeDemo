// src/loop.cpp
#include "init.h"
#include "loop.h"
#include "mainProgramInc.h"
#include "core/timeMng.h"

namespace CubeDemo {
extern std::vector<Model*> ModelPointers;

void MainLoop(WIN, CAM) {
    while (!WindowMng::ShouldClose()) {
        BeginFrame(camera);    // 开始帧
        HandleInput(window);    // 输入管理
        if (!InputHandler::isGamePaused) { UpdateModels(); }    // 模型渲染
        HandleWindowSettings(window);    // 窗口输入设置
        RenderScene(window, camera);
        EndFrameHandling(window);
    }
}

// 开始帧
void BeginFrame(CAM) {
    Renderer::BeginFrame();
    TimeMng::Update();
    UIMng::RenderLoop(WindowMng::GetWindow(), *camera);
}

// 输入管理
void HandleInput(WIN) {
    static float lastEscPressTime = 0.0f;
    const float currentTime = glfwGetTime();
    
    if ((currentTime - lastEscPressTime) > InputHandler::escCoolDown) {
        lastEscPressTime = currentTime;
        if (!InputHandler::isGamePaused) {
            InputHandler::PauseTheGame(window);
        } else if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            InputHandler::ResumeTheGame(window);
        }
    }

    if (!InputHandler::isGamePaused) {
        InputHandler::ProcessKeyboard(window, TimeMng::DeltaTime());
    }
}

// 模型变换(如旋转)
void UpdateModels() {

}

// 输入窗口设置
void HandleWindowSettings(WIN) {
    WindowMng::UpdateWindowSize(window);    // 更新窗口尺寸
    float aspectRatio = CalculateAspectRatio(
        WindowMng::s_WindowWidth, 
        WindowMng::s_WindowHeight
    );  // 计算纵横比
    WindowMng::FullscreenTrigger(window);   // 全屏 
}

// 计算纵横比
float CalculateAspectRatio(int w, int h) {
    return (h == 0) ? 1.0f : static_cast<float>(w) / h;
}

// 渲染场景
void RenderScene(WIN, CAM) {
    WindowMng::UpdateWindowSize(window);

    const float aspectRatio = CalculateAspectRatio(WindowMng::s_WindowWidth, WindowMng::s_WindowHeight);

    // 
    Shader modelShader(
        "../res/shaders/vertex/core/model.glsl",
        "../res/shaders/fragment/core/model.glsl"
    );
    modelShader.Use();

    modelShader.ApplyCamera(*camera, CalculateAspectRatio(WindowMng::s_WindowWidth, WindowMng::s_WindowHeight));

    for(auto* thisModel : ModelPointers) {
        thisModel->Draw(modelShader);

    }
}

void EndFrameHandling(WIN) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}


}