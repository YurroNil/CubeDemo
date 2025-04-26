// src/loop.cpp
#include "init.h"
#include "loop.h"
#include "mainProgramInc.h"
#include "core/time.h"
#include <future>

namespace CubeDemo {
extern std::vector<Model*> MODEL_POINTERS;
extern Shader* MODEL_SHADER;
constexpr int MAX_TASKS_PER_FRAME = 50; // 新增帧任务限制



void MainLoop(WIN, CAM) {


    while (!Window::ShouldClose()) {
        static int var = 0;
        TaskQueue::ProcTasks(); // 任务处理

         // 优化帧率控制（使用精确计时）
        static auto lastTime = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto delta = now - lastTime;
        
        if(delta < std::chrono::microseconds(16667)) { // 60FPS
            std::this_thread::sleep_for(
                std::chrono::microseconds(16667) - delta
            );
        }
        lastTime = std::chrono::steady_clock::now();


        BeginFrame(camera);    // 开始帧
        HandleInput(window);    // 输入管理
        if (!Inputs::isGamePaused) { UpdateModels(); }    // 模型渲染
        HandleWindowSettings(window);    // 窗口输入设置
        RenderScene(window, camera);
        EndFrameHandling(window);

        std::cout << "当前活动纹理数: " << TextureLoader::s_TexturePool.size() << " 活动模型数: " << MODEL_POINTERS.size() << "\n";
        var++;
        std::cout << "[断点] 当前状态是第" << var << "次循环" << std::endl;
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
        if (thisModel->IsReady() &&
            camera->isSphereVisible(thisModel->bounds.Center, thisModel->bounds.Rad)
        ) { thisModel->Draw(*MODEL_SHADER); }
    }

}   // RenderScene

void EndFrameHandling(WIN) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}

}