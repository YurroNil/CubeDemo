// include/core/timeManager.h

#pragma once

class TimeManager {
public:
    static void Update();
    static float FPS();
    static float DeltaTime();
    
private:
    inline static float s_LastTime = 0.0f;
    inline static float s_DeltaTime = 0.0f;
};