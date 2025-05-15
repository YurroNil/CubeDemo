// src/loop.cpp
#include "init.h"
#include "loop.h"
#include "main/includes.h"
#include "core/time.h"
#include <future>

namespace CubeDemo {
extern std::vector<Model*> MODEL_POINTERS; extern Shader* MODEL_SHADER;
extern bool DEBUG_LOD_MODE;

/* ---------------- 程序主循环 -------------- */
void MainLoop(WIN, CAM) {

    while (!Window::ShouldClose()) {
        static int var = 0; int processed = 0;

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
        render_scene(window, camera);    

        /* 结束帧 */
        end_frame_handling(window);
    }

} /* ---------------- 程序主循环 -------------- */

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
    Window::UpdateWindowSize(window);    // 更新窗口尺寸
    Window::FullscreenTrigger(window);   // 全屏 
}

/* <------------ 渲  染  循  环 ------------> */
void render_scene(WIN, CAM) {

    Window::UpdateWindowSize(window);

/* ------应用模型着色器------ */
    MODEL_SHADER->Use();

    // 到摄像机
    MODEL_SHADER->ApplyCamera(*camera, Window::GetAspectRatio());

    // 到模型
    for (auto* model : MODEL_POINTERS) {
        if (!model->IsReady()) {
            std::cout << "[Render] 模型未就绪: " << model << std::endl;
            continue;
        }

        // const float distance = glm::distance(model->bounds.Center, camera->Position);
        
        // 视椎体裁剪判断
        if (model->IsReady() &&
            camera->isSphereVisible(model->bounds.Center, model->bounds.Rad)
        ) {
            model->DrawCall(DEBUG_LOD_MODE, *MODEL_SHADER, camera->Position);
        }
    }
}   // RenderScene

// 结束帧
void end_frame_handling(WIN) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}

}   // Namespace CubeDemo
