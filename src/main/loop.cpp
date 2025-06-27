// src/main/loop.cpp
#include "pch.h"
#include "main/loop.h"
#include "threads/task_queue.h"
#include "managers/uiMng.h"

// 外部变量声明
namespace CubeDemo {
    extern SceneMng* SCENE_MNG;
    extern ShadowMap* SHADOW_MAP;
}

namespace CubeDemo {

/* ---------------- 程序主循环 -------------- */

void MainLoop(GLFWwindow* window, Camera* camera) {

    /* 渲染循环初始化 */
    UIMng::RenderInit();

    // 开始主循环
    while (!WINDOW::ShouldClose()) {
        int processed = 0;

        /* 任务处理 */
        TaskQueue::ProcTasks(processed);

        /* 开始帧 */
        begin_frame(camera);

        /* 输入管理 */
        handle_input(window, camera);

        /* 窗口输入设置 */
        handle_window_settings(window);

        /* --- 分辨率检查 --- */
        // 更新窗口尺寸并检查分辨率
        WINDOW::UpdateWinSize(window);

        

        // 只有分辨率足够时才渲染主要部分
        if (WINDOW::IsResolutionSupported()) {
            /* 渲染UI面板 */
            UIMng::RenderLoop(window, camera);

            /* 模型状态更新 */
            if (!INPUTS::s_isGamePaused) update_models();

            /* 渲染场景 */
            render_scene(window, camera, SHADOW_MAP);
        } else {
            // 否则渲染分辨率错误的界面
            UIMng::RenderResolutionError();
        }

        /* 结束帧 */
        end_frame_handling(window);
    }
}

}   // Namespace CubeDemo
