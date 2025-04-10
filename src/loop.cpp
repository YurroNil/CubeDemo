// src/loop.cpp

#include "loop.h"
#include "mainProgramInc.h"
#include "core/timeMng.h"

namespace CubeDemo {

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

// 模型更新
void UpdateModels() {

    // 立方体
    static float rotation = 0.0f;
    rotation += TimeMng::DeltaTime() * 50.0f;
    ModelMng::SetRotation("cube", vec3(0.0f, rotation, 0.0f));

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
    
    ModelMng::Render("cube", *camera, aspectRatio);
}

void EndFrameHandling(WIN) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}


}