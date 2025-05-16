// include/loop.h

#pragma once
#include "kits/glfw.h"
#include "core/camera.h"
#include "scenes/mainScene.h"

#define WIN GLFWwindow* window
#define CAM Camera* camera

using Scene = CubeDemo::Scenes::MainScene;

namespace CubeDemo {

    // 程序主循环模块
    void MainLoop(WIN, CAM);
    void begin_frame(CAM);
    void handle_input(WIN);
    void update_models();
    void handle_window_settings(WIN);
    void render_scene(WIN, CAM, Scene& scene_inst);
    void end_frame_handling(WIN);

}
