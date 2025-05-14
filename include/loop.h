// include/loop.h

#pragma once
#include "kits/glfw.h"
#include "core/camera.h"

#define WIN GLFWwindow* window
#define CAM Camera* camera

namespace CubeDemo {
    
    // 程序主循环模块
    void MainLoop(WIN, CAM);
    void begin_frame(CAM);
    void handle_input(WIN);
    void update_models();
    void handle_window_settings(WIN);
    void render_scene(WIN, CAM);
    void end_frame_handling(WIN);

}