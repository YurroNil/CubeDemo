// include/core/camera.h

#pragma once
#include "utils/glmKits.h"

namespace CubeDemo {

class Camera {
public:
    // 相机属性
    vec3 Position; vec3 Front; vec3 Up; vec3 Right; vec3 WorldUp;
    // 欧拉角
    float Yaw; float Pitch;
    // 相机选项
    float MovementSpeed; float MouseSensitivity; float Zoom;
    Camera(
        vec3 position,
        vec3 up,
        float yaw, 
        float pitch
    );
    mat4 GetViewMatrix() const;     // 获取视图矩阵
    void ProcessKeyboard(int direction, float deltaTime);
    void ProcessMouseMovement(float xoffset, float yoffset);
    void ProcessMouseScroll(float yoffset);
    void Jump(float velocity);
    // 保存摄像机对象的指针
    static void SaveCamera(Camera* c);
    // 获取摄像机对象的指针
    static Camera* GetCamera();
    static void Delete(Camera* c);

private:
    void updateCameraVectors();
    inline static Camera* SaveCameraPtr = nullptr;
};


}