#include "core/keyMapper.h"
namespace CubeDemo {
// KeyMapper类说明：
// 该类用于管理键盘输入与游戏动作的绑定，支持组合键和持续按键检测
// 当前实现支持：基础按键绑定、修饰键组合、持续触发动作
// 注意：需在主循环中持续调用ProcessInput以检测输入状态

// 存储动作绑定的结构体（假设在头文件中定义）
struct ActionBinding {
    int key;                // 绑定的按键码
    int modifiers;          // 所需的修饰键组合（位掩码）
    std::function<void(float)> action;  // 执行的操作（带速度参数）
};

// 注册按键绑定（添加新的输入动作）
/**
 * @param key 要绑定的按键码（GLFW_KEY_*）
 * @param modifiers 需要的修饰键组合（位掩码，如GLFW_MOD_CONTROL | GLFW_MOD_SHIFT）
 * @param action 要执行的操作函数（接受速度参数）
 */
void KeyMapper::RegisterAction(int key, int modifiers, std::function<void(float)> action) {
    // 将新的绑定添加到动作列表
    m_Actions.push_back({key, modifiers, action});
}

// 输入处理主函数（需每帧调用）
/**
 * @param window GLFW窗口指针
 * @param velocity 当前帧的速度系数（用于控制动作强度）
 * @note 应在主渲染循环中调用，处理所有已注册的输入绑定
 */
void KeyMapper::ProcessInput(GLFWwindow* window, float velocity) {
    // 遍历所有已注册的动作绑定
    for (const auto& action : m_Actions) {
        // 检测按键是否被按下
        bool keyPressed = glfwGetKey(window, action.key) == GLFW_PRESS;
        
        // 检测修饰键是否匹配
        bool modifiersMatch = CheckModifiers(window, action.modifiers);
        
        // 当按键按下且修饰键匹配时执行动作
        if (keyPressed && modifiersMatch) {
            action.action(velocity);  // 传入速度系数控制动作幅度
        }
    }
}

// 修饰键状态检测（内部使用）
/**
 * @param requiredMods 需要的修饰键组合（位掩码）
 * @return true如果所有需要的修饰键都被按下，false否则
 * @note 使用位运算处理多个修饰键的组合检测
 */
bool KeyMapper::CheckModifiers(GLFWwindow* window, int requiredMods) {
    // 逻辑解释：
    // 对于每个修饰键，如果requiredMods中不需要该键（对应位为0），或者该键被按下，则条件成立
    // 使用德摩根定律将多个&&条件转换为更易读的格式
    return (
        (!(requiredMods & GLFW_MOD_SHIFT)    || IsShiftPressed(window))    &&  // 不需要Shift或Shift被按下
        (!(requiredMods & GLFW_MOD_CONTROL)  || IsControlPressed(window))  &&  // 同理Control
        (!(requiredMods & GLFW_MOD_ALT)      || IsAltPressed(window))      &&
        (!(requiredMods & GLFW_MOD_SUPER)    || IsSuperPressed(window))
    );
}

// 修饰键检测辅助函数（处理左右两侧修饰键）
// 以下四个函数检测特定修饰键的按下状态（考虑左右两侧）
bool KeyMapper::IsShiftPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
}

bool KeyMapper::IsControlPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
}
// 点击Alt键
bool KeyMapper::IsAltPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_ALT)  == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
}

bool KeyMapper::IsSuperPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_SUPER)  == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS;
}

// 初始化默认输入绑定
/**
 * @param keyMapper 要初始化的KeyMapper实例
 * @param window GLFW窗口指针
 * @param camera 相机对象指针（用于移动控制）
 * @note 演示了多种绑定方式：基础按键、组合键、lambda表达式使用
 */
void KeyMapper::Init(KeyMapper& keyMapper, GLFWwindow* window, Camera* camera) {
    // 注册ESC退出（无修饰键）
    keyMapper.RegisterAction(GLFW_KEY_ESCAPE, 0, [window](float) {
        glfwSetWindowShouldClose(window, true);  // 设置窗口关闭标志
    });

    // 注册基础移动控制（使用lambda封装重复逻辑）
    // 定义通用移动注册函数
    auto registerMovement = [&](int key, auto direction) {
        keyMapper.RegisterAction(key, 0, [camera, direction](float velocity) {
            // 根据方向函数获取移动方向，乘以速度系数
            camera->Position += direction(*camera) * velocity;
        });
    };

    // 绑定WASD移动（前后左右）
    registerMovement(GLFW_KEY_W, [](Camera& c) { return c.Front; });    // 前进
    registerMovement(GLFW_KEY_S, [](Camera& c) { return -c.Front; });   // 后退
    registerMovement(GLFW_KEY_A, [](Camera& c) { return -c.Right; });   // 左移
    registerMovement(GLFW_KEY_D, [](Camera& c) { return c.Right; });    // 右移

    // 垂直移动绑定
    keyMapper.RegisterAction(GLFW_KEY_SPACE, 0, [camera](float velocity) {
        camera->Position.y += velocity;  // 空格键上升
    });
    keyMapper.RegisterAction(GLFW_KEY_LEFT_SHIFT, 0, [camera](float velocity) {
        camera->Position.y -= velocity;  // Shift键下降
    });

    // 组合键示例：Ctrl+W加速前进（3倍速度）
    keyMapper.RegisterAction(GLFW_KEY_W, GLFW_MOD_CONTROL, [camera](float velocity) {
        camera->Position += camera->Front * (velocity * 3.0f);  // 速度乘以3
    });
}


}