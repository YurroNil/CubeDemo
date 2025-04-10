// include/core/timeMng.h

#pragma once
namespace CubeDemo {


class TimeMng {
public:
    static void Update();
    static int FPS();
    static float DeltaTime();
    
private:
    inline static float s_LastTime = 0.0f;
    inline static float s_DeltaTime = 0.0f;
};


}