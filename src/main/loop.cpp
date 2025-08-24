// src/main/loop.cpp
#include "pch.h"
#include "main/loop.h"
#include "threads/task_queue.h"
#include "managers/ui.h"
#include "ui/screens/loading.h"
#include "ui/main_menu/panel.h"

namespace CubeDemo {

// 外部变量声明
extern SceneMng* SCENE_MNG;
extern bool DEBUG_ASYNC_MODE, RAY_TRACING_ENABLED, RT_DEBUG;

/* ---------------- 程序主循环 -------------- */

void MainLoop(GLFWwindow* window, Camera* camera) {

    // 开始主循环
    while (!WINDOW::ShouldClose()) {
        int processed = 0;

        /* 任务处理 */
        TaskQueue::ProcTasks(processed);

        /* 开始帧 */
        begin_frame(camera);

        /* 输入管理 */
        if(!UI::MainMenuPanel::s_isMainMenuPhase) handle_input(window, camera);

        /* 窗口输入设置 */
        handle_window_settings(window);

        // 更新窗口尺寸并检查分辨率
        WINDOW::UpdateWinSize(window);

        /* 渲染判断 */
        rendering_judgment(window, camera);

        /* 结束帧 */
        end_frame_handling(window);
    }
}

// 渲染内容的判断
void rendering_judgment(GLFWwindow* window, Camera* camera) {

    // 是否处于分辨率不支持的状态
    if (!WINDOW::IsResolutionSupported()) {
        // 渲染分辨率错误的界面
        UIMng::RenderResolutionError();
        return;
    }
    // 是否处于加载状态
    if(UI::LoadingScreen::s_isLoading) {
        // 渲染加载画面
        UI::LoadingScreen::Render(DEBUG_ASYNC_MODE);
        return;
    }

    // 是否处于主菜单阶段
    if (UI::MainMenuPanel::s_isMainMenuPhase) {
        UI::MainMenuPanel::Render();
        return;
    }

    /* ------若以上情况都不是，那么正常渲染画面------ */

    /* 渲染UI面板 */
    UIMng::RenderLoop(window, camera);

    /* 模型状态更新 */
    if (!INPUTS::s_isGamePaused) update_models();

    /* 渲染场景 */
    render_scene(window, camera);
}

/* <------------ 渲  染  循  环 ------------> */
void render_scene(GLFWwindow* window, Camera* camera) {
    // GLSL渲染路径
    if(auto* scene = SCENE_MNG->GetCurrentScene()) scene->Render(window, camera);
}

// 模型变换(如旋转)
void update_models() {}

// 输入管理
void handle_input(GLFWwindow* window, Camera* camera) {
    INPUTS::ProcPanelKeys(window);
    INPUTS::ProcCameraKeys(window, camera, TIME::GetDeltaTime());
}
// 开始帧
void begin_frame(Camera* camera) {
    Renderer::BeginFrame();
    TIME::Update();
}

// 结束帧
void end_frame_handling(GLFWwindow* window) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}

// 输入窗口设置
void handle_window_settings(GLFWwindow* window) {
    WINDOW::UpdateWinSize(window);       // 更新窗口尺寸
    WINDOW::FullscreenTrigger(window);   // 全屏
}
}   // Namespace CubeDemo
