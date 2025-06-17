// include/core/inputs.h
#pragma once

namespace CubeDemo {
class Camera;
class Inputs {
public:
    static void Init(Camera* camera);
    static void ProcKeyboard(GLFWwindow* window, float delta_time);
    inline static bool showDebugInfo = false;
    inline static float frameTime = 0.0f;

    // 静态回调实现
    static void MouseCallback(double xpos, double ypos);
    static void ScrollCallback(double yoffset);

    inline static bool isGamePaused = false;
    static void ResumeTheGame(GLFWwindow* window);
    static float lastEscPressTime;  // 上次按下ESC的时间
    static constexpr float s_EscCoolDown = 0.1f;  // 冷却时间100ms
    static void PauseTheGame(GLFWwindow* window);
    static bool s_isDebugVisible;   // 调试信息面板状态

    static void isEscPressed(GLFWwindow* window);
    
    static void HandleModelPlacement(GLFWwindow* window);

     // 编辑模式状态
    inline static bool s_isEditMode = false;
    static void ToggleEditMode(GLFWwindow* window);
    
    // 放置状态
    inline static string s_PlacementModelID = "";
    static void SetPlacementModel(const string& model_id);
    static void ClearPlacementModel();

private:
    inline static Camera* s_Camera = nullptr;
    inline static float s_LastX = 0.0f;
    inline static float s_LastY = 0.0f;
    inline static bool s_FirstMouse = true;
    inline static bool s_AltPressed = false;
    inline static float s_LastToggleTime = 0.0f;
    inline static constexpr float s_ToggleCD = 0.3f;
};
}