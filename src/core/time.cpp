// src/core/time.cpp
#include "pch.h"
#include "core/time.h"

namespace CubeDemo {

namespace {
    float s_LastUpdate = 0.0f;
    int s_CachedFPS = 0;
    int s_FrameCounter = 0;
    const float UPDATE_INTERVAL = 2.0f; // 2秒更新间隔
}

void Time::Update() {
    static float last_frame = 0.0f;
    float current_frame = glfwGetTime();
    float delta_time = current_frame - last_frame;
    last_frame = current_frame;
    s_DeltaTime = delta_time;

    // 每帧计数器累加
    s_FrameCounter++;

    // 检查是否达到更新间隔
    if (current_frame - s_LastUpdate >= UPDATE_INTERVAL) {
        float elapsed = current_frame - s_LastUpdate;
        s_CachedFPS = static_cast<int>(s_FrameCounter / elapsed);
        s_LastUpdate = current_frame;
        s_FrameCounter = 0;
    }
}

int Time::FPS() { return s_CachedFPS; }

float Time::DeltaTime() { return s_DeltaTime; }

}