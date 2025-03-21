// include/core/fpsCounter.h

#pragma once

class FPSCounter {
public:
    static void Update();
    static float GetFrameTime();
    static float GetFPS();

private:
    inline static float s_FrameTime = 0.0f;
    inline static float s_LastTime = 0.0f;
};