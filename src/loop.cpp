// src/loop.cpp
#include "init.h"
#include "loop.h"
#include "main/includes.h"
#include "core/time.h"
#include <future>

namespace CubeDemo {

extern std::vector<Model*> MODEL_POINTERS;
extern Shader* MODEL_SHADER;
extern bool DEBUG_LOD_MODE;

/* ---------------- 程序主循环 -------------- */
void MainLoop(WIN, CAM) {

    Scene scene_inst;
    scene_inst.Init();

    while (!Window::ShouldClose()) {
        int processed = 0;

        /* 任务处理 */
        TaskQueue::ProcTasks(processed);

        /* 开始帧 */
        begin_frame(camera);

        /* 输入管理 */
        handle_input(window);

        /* 模型状态更新 */
        if (!Inputs::isGamePaused) update_models();

        /* 窗口输入设置 */
        handle_window_settings(window);

        /* 渲染场景 */
        render_scene(window, camera, scene_inst);    

        /* 结束帧 */
        end_frame_handling(window);
    }
}

// 开始帧
void begin_frame(CAM) {
    Renderer::BeginFrame();
    Time::Update();
    UIMng::RenderLoop(Window::GetWindow(), *camera);
}

// 输入管理
void handle_input(WIN) {
    Inputs::isEscPressed(window);
    if (!Inputs::isGamePaused) Inputs::ProcKeyboard(window, Time::DeltaTime());
}

// 模型变换(如旋转)
void update_models() {
}   // UpdateModels

// 输入窗口设置
void handle_window_settings(WIN) {
    Window::UpdateWinSize(window);    // 更新窗口尺寸
    Window::FullscreenTrigger(window);   // 全屏 
}

/* <------------ 渲  染  循  环 ------------> */
void render_scene(WIN, CAM, Scene& scene_inst) {

    // 阴影渲染阶段
    scene_inst.RenderShadow(camera);

    // 主渲染阶段
    scene_inst.RenderMainPass(window, camera);

}   // RenderScene

// 结束帧
void end_frame_handling(WIN) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}

}   // Namespace CubeDemo
