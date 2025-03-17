#pragma once
#include <tplib/GLFW/glfw3.h>

class windowManager {
public:
    static void ToggleFullscreen(GLFWwindow* window);

private:
    inline static bool s_IsFullscreen = false;
    inline static int s_WindowPosX = 0;
    inline static int s_WindowPosY = 0;
    inline static int s_WindowWidth = 800;
    inline static int s_WindowHeight = 600;
};
