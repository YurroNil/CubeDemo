// include/core/inputs.h
#pragma once

namespace CubeDemo {
class INPUTS {
    // 状态管理结构体
    struct MouseState {
        double lastX, lastY, savedX, savedY;
        bool firstMove, isVsble;

        MouseState(): 
        lastX(0.0), lastY(0.0),
        savedX(0.0), savedY(0.0),
        firstMove(true),
        isVsble(false) {}
    };

    // 静态私有成员
    inline static MouseState m_Mouse;
    inline static float m_LastToggleTime = 0.0f;
    inline static bool m_AltPressed{false}, m_needCameraReset{false};

    static std::unordered_map<int, std::function<void()>> s_PanelkeyMap;
    static std::unordered_map<int, std::function<void(Camera* camera, float velocity)>> s_CamerakeyMap;
    static std::unordered_map<int, bool> s_KeyState;

    // 常量
    static constexpr float TOGGLE_CD = 0.3f;
    static constexpr float ESC_CD = 0.1f;

    // 私有方法
    static void UpdateCursorMode(GLFWwindow* window);
    static bool isOpeningPanel();
    
public:
        // 公有状态
    inline static bool 
        s_isPresetVsble = false, 
        s_isGamePaused = false,
        s_isDebugVsble = false,
        s_isEditMode = false;

    // 核心方法
    static void ProcPanelKeys(GLFWwindow* window);
    static void ProcCameraKeys(GLFWwindow* window, Camera* camera, float deltaTime);
    static void SetPaused(GLFWwindow* window, bool paused);

    // 回调函数
    static void MouseCallback(double xpos, double ypos);
    static void ScrollCallback(double yoffset);
};
} // namespace CubeDemo
