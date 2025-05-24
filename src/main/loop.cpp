// src/main/loop.cpp

#include "main/loop.h"
#include "core/inputs.h"

namespace CubeDemo {

/* ---------------- 程序主循环 -------------- */

void MainLoop(GLFWwindow* window, Camera* camera) {

    Scene scene_inst; Light light;

    // 设置场景为默认场景
    scene_inst.Current = SceneID::DEFAULT;

    // 执行场景初始化
    scene_inst.Init(light);

    // 创建阴影(静态保持阴影贴图)
    ShadowMap* shadow_map = new ShadowMap(2048, 2048);
    shadow_map->CreateShader();

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
        render_scene(window, camera, scene_inst, light, shadow_map);
        
        /* 结束帧 */
        end_frame_handling(window);
    }

    scene_inst.CleanAllScenes(light);
}

}   // Namespace CubeDemo
