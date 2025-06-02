// src/main/loop.cpp

#include "main/loop.h"
#include "core/inputs.h"
#include "threads/taskQueue.h"

// 外部变量声明
namespace CubeDemo {
    extern Scene* SCENE_INST;
    extern ShadowMap* SHADOW_MAP;
}

namespace CubeDemo {

/* ---------------- 程序主循环 -------------- */

void MainLoop(GLFWwindow* window, Camera* camera) {

    Light light;

    // 场景管理器初始化
    SCENE_INST->Init(light);

    // 开始主循环
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
        render_scene(window, camera, SCENE_INST, light, SHADOW_MAP);
        
        /* 结束帧 */
        end_frame_handling(window);
    }

    // 清理场景资源
    SCENE_INST->CleanAllScenes(light);
}

}   // Namespace CubeDemo
