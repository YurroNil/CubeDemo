// src/core/time.cpp

#include "core/time.h"
#include "utils/glfwKits.h"


namespace CubeDemo {

namespace {
    float s_LastUpdate = 0.0f;
    int s_CachedFPS = 0;
    int s_FrameCounter = 0;
    const float UPDATE_INTERVAL = 2.0f; // 2秒更新间隔
}

void Time::Update() {
    static float lastFrame = 0.0f;
    float currentFrame = glfwGetTime();
    float deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    s_DeltaTime = deltaTime;

    // 每帧计数器累加
    s_FrameCounter++;

    // 检查是否达到更新间隔
    if (currentFrame - s_LastUpdate >= UPDATE_INTERVAL) {
        float elapsed = currentFrame - s_LastUpdate;
        s_CachedFPS = static_cast<int>(s_FrameCounter / elapsed);
        s_LastUpdate = currentFrame;
        s_FrameCounter = 0;
    }
}

int Time::FPS() { return s_CachedFPS; }

float Time::DeltaTime() { return s_DeltaTime; }

}