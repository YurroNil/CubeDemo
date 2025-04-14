// include/core/camera.h

#pragma once
#include "utils/glmKits.h"

namespace CubeDemo {

class Camera {
public:

    Camera(vec3 position, vec3 up, float yaw, float pitch);

    // 相机属性
    vec3 Position; vec3 Front; vec3 Up; vec3 Right; vec3 WorldUp;
    // 欧拉角
    float Yaw; float Pitch;
    // 相机选项
    float MovementSpeed; float MouseSensitivity; float Zoom;
    
    mat4 GetViewMatrix() const;
    mat4 GetProjectionMatrix(float aspect) const;
    void ProcessKeyboard(int direction, float deltaTime);
    void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch);
    void ProcessMouseScroll(float yoffset);
    void Jump(float velocity);
    // 保存摄像机对象的指针
    static void SaveCamera(Camera* c);
    // 获取摄像机对象的指针
    static Camera* GetCamera();
    static void Delete(Camera* c);

    float NearPlane = 0.1f;
    float FarPlane = 100.0f;

    struct FrustumPlane {
        vec3 normal;
        vec3 distance;
    };

    struct Frustum {
        FrustumPlane planes[6]; // 顺序：左、右、下、上、近、远
    };

    Frustum GetFrustum(float aspect) const;
    bool CheckSphereVisibility(const vec3& center, float radius) const;

private:
    void UpdateCameraVectors();
    inline static Camera* SaveCameraPtr = nullptr;
};


}