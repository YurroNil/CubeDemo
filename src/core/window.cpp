// src/core/window.cpp
#include "pch.h"
#include "core/inputs.h"
#include "threads/task_queue.h"

namespace CubeDemo {

void Window::Init(int width, int height, const char* title) {

    // 在初始化时捕获主线程ID
    TaskQueue::s_MainThreadId = std::this_thread::get_id();

    // 初始化窗口
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    m_Window = glfwCreateWindow(width, height, "Cube Demo", NULL, NULL);
    if (!m_Window) { glfwTerminate(); throw std::runtime_error("[Error] 窗口创建失败"); }

    // 设置OpenGL上下文
    glfwMakeContextCurrent(m_Window);
    if (gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0) {
        throw std::runtime_error("[Error] GLAD初始化失败");
        glfwDestroyWindow(m_Window);
        glfwTerminate();
    }
    
    int win_width, win_height; glfwGetFramebufferSize(m_Window, &win_width, &win_height);
    m_InitMouseX = win_width / 2.0f; m_InitMouseY = win_height / 2.0f;

    // 设置GLFW回调
    glfwSetCursorPosCallback(m_Window, [](GLFWwindow* w, double x, double y) { Inputs::MouseCallback(x, y); });
    glfwSetScrollCallback(m_Window, [](GLFWwindow* w, double x, double y) { Inputs::ScrollCallback(y); });
    glfwSetInputMode(m_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // 添加窗口大小回调来更新窗口
    glfwSetFramebufferSizeCallback(m_Window, [](GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }); 

    // 请求32位深度缓冲
    glfwWindowHint(GLFW_DEPTH_BITS, 32);

/* 
关于深度缓冲的说明 (重要!!)

    OpenGL深度缓冲默认的值是24位。若发生深度缓冲精度不足的情况，则会发生以下诡异的现象: 摄像机远离模型时，远处贴图似乎“逼近”摄像机, 并最终穿过近处模型.

    这是典型的`深度缓冲冲突(Z-fighting)`问题，根本原因在于 深度缓冲（Z-buffer）的精度不足，尤其是在处理远近差异极大的场景时.

    问题根源：深度缓冲精度限制
    1. 非线性深度分布

    透视投影中，深度值（Z值）在投影矩阵变换后是非线性分布的（通常使用倒数关系 1/z 存储）。
    精度分布不均：近处物体的深度值精度高，远处物体的深度值精度极低（深度值集中在近平面附近）。


    2. 深度值量化误差

    深度缓冲通常使用 24位浮点数（范围 [0, 1]）。
    当摄像机远离物体时，物体在深度缓冲中的深度值差异极小（例如 0.999999 vs 0.999998），超出浮点数精度范围，导致深度比较失败。

    结果：
    相邻像素的深度值因精度不足被判定为“相等”，OpenGL无法确定渲染顺序。
    模型和贴图交替渲染，产生闪烁/穿透现象。

    解决方案除了提升深度缓冲精度以外，更常见的解决方案是使用对数深度. 对数深度是常用的解决深度冲突的方法，在常见的渲染引擎中都有涉及，对数深度分布可以显著改善深度缓冲区的精度问题，尤其是在大规模地形渲染中。对数深度缓冲区通过非线性地分布深度值，使得深度值在近处有较高的精度，而在远处也有更均匀的分布。

    为了实现对数深度分布，通常我们会在深度值通过投影矩阵转换之后，对其进行修改。具体的做法是对深度值应用对数变换。这里有一个常见的实现步骤：

    计算投影矩阵中的深度值：在标准的深度测试中，深度值在 [0, 1] 范围内，经过投影矩阵转换后的深度值也在这个范围内。
    应用对数变换：对投影矩阵转换后的深度值应用对数变换。假设 z 是经过投影矩阵转换后的深度值，我们可以通过以下公式进行对数变换.

    在opengl中，深度范围[-1,1]，转换公式为:
    z_log = [ 2*log(C*w + 1) / log(C*Far + 1) ] * W

    其中，C 是一个常数，用来确定相机附近的分辨率，一般取1就可以，而乘以 W 是为了提前取消后续管线中的隐式除法。

*/

    // 输出OpenGL版本
    std::cout << "OpenGL版本: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GPU: " << glGetString(GL_RENDERER) << std::endl;
}

bool Window::ShouldClose() { return glfwWindowShouldClose(m_Window); }

void Window::ToggleFullscreen(GLFWwindow* window) {
    if(!window) return; // 防止空指针
    if (!m_IsFullscreen) {
        // 保存窗口位置和尺寸
        UpdateWinSize(window); UpdateWindowPos(window);
        // 切换到全屏
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    } else {
        // 恢复窗口
        glfwSetWindowMonitor(window, nullptr, m_WinPosX, m_WinPosY, m_Width, m_Height, GLFW_DONT_CARE);
    }
    m_IsFullscreen = !m_IsFullscreen;
}

void Window::FullscreenTrigger(GLFWwindow* window) {
    static bool f11_last_state = false;
    bool f11_current_state = glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS;
    if (f11_current_state && !f11_last_state) { ToggleFullscreen(window); }
    f11_last_state = f11_current_state;
}

// Setters
void Window::UpdateWinSize(GLFWwindow* window) { glfwGetWindowSize(window, &m_Width, &m_Height); }

void Window::UpdateWindowPos(GLFWwindow* window) { glfwGetWindowPos(window, &m_WinPosX, &m_WinPosY); }

// Getters
float Window::GetAspectRatio() { return (m_Height == 0) ? 1.0f : static_cast<float>(m_Width) / m_Height; }

GLFWwindow* Window::GetWindow() { return m_Window; }
float Window::GetInitMouseX() { return m_InitMouseX; }
float Window::GetInitMouseY() { return m_InitMouseY; }
const int Window::GetWidth() { return m_Width; };
const int Window::GetHeight() { return m_Height; };

}   // namespace CubeDemo
