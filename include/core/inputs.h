// include/core/inputs.h
#pragma once

namespace CubeDemo {
class Inputs {
    // 友元类与函数声明
    friend class Window; friend void handle_input(GLFWwindow* window);

    // 静态私有方法
    static void MouseCallback(double xpos, double ypos);
    static void ScrollCallback(double yoffset);
    static void PauseTheGame(GLFWwindow* window);
    static void isEscPressed(GLFWwindow* window);
    static bool isCameraEnabled();

    // Togglers
    static void ToggleEditMode(GLFWwindow* window);
    static void TogglePresetlib(GLFWwindow* window);

    inline static Camera* m_Camera = nullptr;
    inline static float m_LastX = 0.0f, m_LastY = 0.0f, m_LastToggleTime = 0.0f;
    inline static bool m_FirstMouse = true, m_AltPressed = false;
    inline static constexpr float m_ToggleCD = 0.3f;

public:
    // 静态成员-布尔型
    inline static bool
        s_isPresetVisible = false, isGamePaused = false,
        s_isDebugVisible = false, s_isEditMode = false;

    // 静态成员-整型/浮点型
    static constexpr float s_EscCoolDown = 0.1f;  // 冷却时间100ms

    // 静态方法
    static void Init(Camera* camera);
    static void ProcKeyboard(GLFWwindow* window, float delta_time);

    static void ResumeTheGame(GLFWwindow* window);
  
};
}