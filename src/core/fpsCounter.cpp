// src/core/fpsCounter.cpp

#include "core/fpsCounter.h"
#include "GLFW/glfw3.h"

void FPSCounter::Update() {
    float currentTime = glfwGetTime();
    s_FrameTime = currentTime - s_LastTime;
    s_LastTime = currentTime;
}

float FPSCounter::GetFrameTime() { return s_FrameTime; }
float FPSCounter::GetFPS() { return 1.0f / s_FrameTime; }
