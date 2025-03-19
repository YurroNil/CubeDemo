#pragma once
#include "Camera.h"
#include "WindowManager.h"
#include <3rd-lib/GLFW/glfw3.h>

class InputHandler {
public:
    static void Initialize(Camera* camera, GLFWwindow* window);
    static void ProcessKeyboard(GLFWwindow* window, float deltaTime);

private:
    // 静态成员
    inline static Camera* s_Camera = nullptr;
    inline static GLFWwindow* s_Window = nullptr;
    inline static float s_LastX = 0.0f;
    inline static float s_LastY = 0.0f;
    inline static bool s_FirstMouse = true;
    inline static bool s_AltPressed = false;

    // 静态回调实现
    static void MouseCallback(GLFWwindow* window, double& xpos, double& ypos);
    static void ScrollCallback(GLFWwindow* window, double& xoffset, double& yoffset);
};