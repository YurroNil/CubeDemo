// include/core/window.h
#pragma once

namespace CubeDemo {

class WINDOW {
public:

    // 静态方法
    static void Init(int width, int height, const char* title);

    static void ToggleFullscreen(GLFWwindow* window);
    static void FullscreenTrigger(GLFWwindow* window);
    static bool ShouldClose();

    static void UpdateWinSize(GLFWwindow* window);
    static void UpdateWindowPos(GLFWwindow* window);

    // Getters
    static GLFWwindow* GetWindow();
    static float GetInitMouseX();
    static float GetInitMouseY();
    static float GetAspectRatio();
    static const int GetWidth();
    static const int GetHeight();

        // 添加分辨率检查方法
    static bool IsResolutionSupported() {
        return m_Width >= 1280 && m_Height >= 720;
    }

private:
    // 静态成员
    inline static int
        m_Width = 800, m_Height = 600,
        m_WinPosX = 0, m_WinPosY = 0;

    inline static GLFWwindow* m_Window = nullptr;
    inline static bool m_IsFullscreen{false}, m_ResolutionError{false};
    inline static float m_InitMouseX{0.0f}, m_InitMouseY{0.0f};
};
}
