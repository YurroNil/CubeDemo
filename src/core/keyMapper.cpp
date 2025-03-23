#include "core/keyMapper.h"

// KeyMapper类暂未时装

void KeyMapper::RegisterAction(int key, int modifiers, std::function<void(float)> action) {

    m_Actions.push_back({key, modifiers, action});
}

void KeyMapper::ProcessInput(GLFWwindow* window, float velocity) {  // *需放入主循环执行
    for (const auto& action : m_Actions) {
        bool keyPressed = glfwGetKey(window, action.key) == GLFW_PRESS;
        bool modifiersMatch = CheckModifiers(window, action.modifiers);
        
        if (keyPressed && modifiersMatch) {
            action.action(velocity);
        }
    }
}

bool KeyMapper::CheckModifiers(GLFWwindow* window, int requiredMods) {
    return (!(requiredMods & GLFW_MOD_SHIFT)    || IsShiftPressed(window)) &&
            (!(requiredMods & GLFW_MOD_CONTROL)  || IsControlPressed(window)) &&
            (!(requiredMods & GLFW_MOD_ALT)      || IsAltPressed(window)) &&
            (!(requiredMods & GLFW_MOD_SUPER)    || IsSuperPressed(window));
}

bool KeyMapper::IsShiftPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
}

bool KeyMapper::IsControlPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
}

bool KeyMapper::IsAltPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
}

bool KeyMapper::IsSuperPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS;
}

// 初始化输入绑定 使用KeyMapper::Init()进行初始化
void KeyMapper::Init(KeyMapper& keyMapper, GLFWwindow* window, Camera* camera) {
    // 注册ESC退出
    keyMapper.RegisterAction(GLFW_KEY_ESCAPE, 0, [window](float) {
        glfwSetWindowShouldClose(window, true);
    });

    // 注册移动控制
    auto registerMovement = [&](int key, auto direction) {
        keyMapper.RegisterAction(key, 0, [camera, direction](float velocity) {
            camera->Position += direction(*camera) * velocity;
        });
    };

    registerMovement(GLFW_KEY_W, [](Camera& c) { return c.Front; });
    registerMovement(GLFW_KEY_S, [](Camera& c) { return -c.Front; });
    registerMovement(GLFW_KEY_A, [](Camera& c) { return -c.Right; });
    registerMovement(GLFW_KEY_D, [](Camera& c) { return c.Right; });

    // 垂直移动
    keyMapper.RegisterAction(GLFW_KEY_SPACE, 0, [camera](float velocity) {
        camera->Position.y += velocity;
    });

    keyMapper.RegisterAction(GLFW_KEY_LEFT_SHIFT, 0, [camera](float velocity) {
        camera->Position.y -= velocity;
    });

    // 组合键：Ctrl+W加速前进
    keyMapper.RegisterAction(GLFW_KEY_W, GLFW_MOD_CONTROL, [camera](float velocity) {
        camera->Position += camera->Front * (velocity * 3.0f); // 3倍速度
    });

}
