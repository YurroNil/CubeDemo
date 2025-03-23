// src/core/timeManager.cpp

#include "core/timeManager.h"
#include "GLFW/glfw3.h"
#include <glm/common.hpp>

void TimeManager::Update() {
    static float lastFrame = 0.0f;
    float currentFrame = glfwGetTime();
    float deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    s_DeltaTime = deltaTime;
}

float TimeManager::FPS() {
    return 1.0f / s_DeltaTime;
}

float TimeManager::DeltaTime() {
    return s_DeltaTime;
}
