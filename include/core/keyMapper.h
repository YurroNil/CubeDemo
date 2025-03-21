#pragma once
#include <GLFW/glfw3.h>
#include <unordered_map>
#include <functional>
#include <vector>
#include "core/camera.h"

class KeyMapper {
public:
    struct KeyAction {
        int key;
        int modifiers; // 使用GLFW_MOD_*组合
        std::function<void(float)> action;
    };
    void RegisterAction(int key, int modifiers, std::function<void(float)> action);
    void ProcessInput(GLFWwindow* window, float velocity);

    static void Init(KeyMapper& keyMapper, GLFWwindow* window, Camera* camera);

private:

    std::vector<KeyAction> m_Actions;

    bool CheckModifiers(GLFWwindow* window, int requiredMods);

    // 辅助函数检查修饰键状态
    static bool IsShiftPressed(GLFWwindow* window); 
    static bool IsControlPressed(GLFWwindow* window);
    static bool IsAltPressed(GLFWwindow* window);
    static bool IsSuperPressed(GLFWwindow* window);

};