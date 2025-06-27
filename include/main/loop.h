// include/main/loop.h
#pragma once
#include "main/rendering.h"

namespace CubeDemo {

    // 程序主循环模块
    void MainLoop(GLFWwindow* window, Camera* camera);

    // 输入处理
    void begin_frame(Camera* camera);
    void end_frame_handling(GLFWwindow* window);
    void handle_input(GLFWwindow* window, Camera* camera);
    void handle_window_settings(GLFWwindow* window);

    // 模型状态更新
    void update_models();

    // 渲染循环
    void render_scene(
        GLFWwindow* window,
        Camera* camera,
        ShadowMap* shadow_map
    );
}
