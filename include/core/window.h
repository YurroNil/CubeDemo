// include/core/window.h

#pragma once
#include "kits/glfw.h"
#include "kits/streams.h"
#include "resources/texture.h"

namespace CubeDemo {

class Window {
public:
    // 静态方法
    static void Init(int width, int height, const char* title);

    static void ToggleFullscreen(GLFWwindow* window);
    static void FullscreenTrigger(GLFWwindow* window);
    static bool ShouldClose();

    static void UpdateWindowSize(GLFWwindow* window);
    static void UpdateWindowPos(GLFWwindow* window);

    // Getters
    static GLFWwindow* GetWindow();
    static float GetInitMouseX();
    static float GetInitMouseY();
    static float GetAspectRatio();
    static const int GetWidth();
    static const int GetHight();

private:
    // 静态成员
    inline static int s_WindowWidth = 800;
    inline static int s_WindowHeight = 600;
    inline static int s_WindowPosX = 0;
    inline static int s_WindowPosY = 0;

    inline static GLFWwindow* s_Window = nullptr;
    inline static bool s_IsFullscreen = false;
    inline static float s_InitMouseX = 0.0f;
    inline static float s_InitMouseY = 0.0f;
};

}