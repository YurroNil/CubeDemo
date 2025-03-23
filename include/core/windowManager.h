// include/core/windowManager.h

#pragma once
#include "GLFW/glfw3.h"

class WindowManager {
public:
    static void Init(int width, int height, const char* title);

    static GLFWwindow* GetWindow() { return s_Window; }

    static float GetInitMouseX() { return s_InitMouseX; }
    static float GetInitMouseY() { return s_InitMouseY; }

    static void ToggleFullscreen(GLFWwindow* window);
    static void FullscreenTrigger(GLFWwindow* window);
    static bool ShouldClose();

private:
    inline static GLFWwindow* s_Window = nullptr;

    inline static bool s_IsFullscreen = false;
    inline static int s_WindowPosX = 0;
    inline static int s_WindowPosY = 0;
    inline static int s_WindowWidth = 800;
    inline static int s_WindowHeight = 600;
    
    inline static float s_InitMouseX = 0.0f;
    inline static float s_InitMouseY = 0.0f;
};
