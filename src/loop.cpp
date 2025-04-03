// src/loop.cpp

#include "loop.h"

void CubeDemo::MainLoop(WIN, CAM) {
    while (!WindowManager::ShouldClose()) {
        BeginFrame(camera);    // 开始帧
        HandleInput(window);    // 输入管理
        if (!InputHandler::isGamePaused) { UpdateModels(); }    // 模型渲染
        HandleWindowSettings(window);    // 窗口输入设置
        RenderScene(window, camera);
        EndFrameHandling(window);
    }
}

// 开始帧
void CubeDemo::BeginFrame(CAM) {
    Renderer::BeginFrame();
    TimeManager::Update();
    UIManager::RenderLoop(WindowManager::GetWindow(), *camera);
}

// 输入管理
void CubeDemo::HandleInput(WIN) {
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
        InputHandler::ProcessKeyboard(window, TimeManager::DeltaTime());
    }
}

// 模型更新
void CubeDemo::UpdateModels() {

    // 立方体
    static float rotation = 0.0f;
    rotation += TimeManager::DeltaTime() * 50.0f;
    ModelManager::SetRotation("cube", vec3(0.0f, rotation, 0.0f));

}

// 输入窗口设置
void CubeDemo::HandleWindowSettings(WIN) {
    WindowManager::UpdateWindowSize(window);

    float aspectRatio = CalculateAspectRatio(
        WindowManager::s_WindowWidth, 
        WindowManager::s_WindowHeight
    );
    WindowManager::FullscreenTrigger(window);
    DebugInfoManager::DisplayDebugInfo(window);
}

// 计算纵横比
float CubeDemo::CalculateAspectRatio(int w, int h) {
    return (h == 0) ? 1.0f : static_cast<float>(w) / h;
}

// 渲染场景
void CubeDemo::RenderScene(WIN, CAM) {
    WindowManager::UpdateWindowSize(window);

    const float aspectRatio = CalculateAspectRatio(WindowManager::s_WindowWidth, WindowManager::s_WindowHeight);
    
    ModelManager::Render("cube", *camera, aspectRatio);
}

void CubeDemo::EndFrameHandling(WIN) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}