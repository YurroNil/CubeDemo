// include/core/time.h
#pragma once
namespace CubeDemo {

class TIME {
    #define i_s inline static
public:
    static void Update();
    static int FPS();
    static float GetDeltaTime();
    
private:
    i_s float s_LastTime = 0.0f;
    i_s float s_DeltaTime = 0.0f;
};
}