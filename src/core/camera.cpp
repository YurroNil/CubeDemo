// src/core/camera.cpp

#include "core/camera.h"
#include "core/window.h"
#include <algorithm>

namespace CubeDemo {


Camera::Camera(vec3 position, vec3 up, float yaw, float pitch)
        // 初始化成员
        : Front(vec3(0.0f, 0.0f, -1.0f)),
          MovementSpeed(2.5f),
          MouseSensitivity(0.1f),
          Zoom(45.0f),
          NearPlane(0.1f),
          FarPlane(100.0f),
          Position(position), WorldUp(up), Yaw(yaw), Pitch(pitch)
    {
        UpdateCameraVectors();
    }

// 获取视图矩阵
mat4 Camera::GetViewMatrix() const {
    return glm::lookAt(Position, Position+Front, Up);
}
// 获取投影矩阵
mat4 Camera::GetProjectionMatrix(float aspect) const {
    return glm::perspective(glm::radians(Zoom), aspect, NearPlane, FarPlane);
}

// 处理键盘输入
void Camera::ProcessKeyboard(int direction, float deltaTime)  {
    float velocity = MovementSpeed * deltaTime;
    switch (direction)
    {
    case 0:
        Position += Front * velocity; // W
        break;
    case 1:
        Position -= Front * velocity; // S
        break;
    case 2:
        Position -= Right * velocity; // A
        break;
    case 3:
        Position += Right * velocity; // D
        break;
    
    default: break;
    }
}

// 处理鼠标移动
void Camera::ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) {

    xoffset *= MouseSensitivity; yoffset *= MouseSensitivity;
    Yaw += xoffset; Pitch += yoffset;

    // 限制俯仰角
    if (Pitch > 89.0f) { Pitch = 89.0f; }
    if (Pitch < -89.0f) { Pitch = -89.0f; }

    UpdateCameraVectors();
}

// 处理滚轮
void Camera::ProcessMouseScroll(float yoffset) {
    Zoom -= yoffset;
    if (Zoom < 1.0f) Zoom = 1.0f;
    if (Zoom > 45.0f) Zoom = 45.0f;
}

//跳跃(未时装)
void Camera::Jump(float velocity) {
    Position.y += velocity;
}


void Camera::UpdateCameraVectors() {
    // 根据欧拉角计算前向量
    vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front);

    // 重新计算右向量和上向量
    Right = glm::normalize(glm::cross(Front, WorldUp));
    Up = glm::normalize(glm::cross(Right, Front));
}

void Camera::SaveCamera(Camera* c) { SaveCameraPtr = c; }

Camera* Camera::GetCamera() { return SaveCameraPtr; }

void Camera::Delete(Camera* c) {
    delete c;
    SaveCameraPtr = nullptr;
}

 // 视锥体相关方法
Camera::Frustum Camera::GetFrustum(float aspectRatio) const {
    Frustum frustum;
    const float halfVSide = FarPlane * tanf(glm::radians(Zoom) * 0.5f);
    const float halfHSide = halfVSide * aspectRatio;
    const vec3 frontMultFar = FarPlane * Front;

    frustum.planes[0] = { glm::normalize(glm::cross(Up, frontMultFar + Right * halfHSide)), Position }; // 左平面
    frustum.planes[1] = { glm::normalize(glm::cross(frontMultFar - Right * halfHSide, Up)), Position }; // 右平面
    frustum.planes[2] = { glm::normalize(glm::cross(Right, frontMultFar - Up * halfVSide)), Position };  // 下平面
    frustum.planes[3] = { glm::normalize(glm::cross(frontMultFar + Up * halfVSide, Right)), Position };  // 上平面
    frustum.planes[4] = { Front, Position + Front * NearPlane }; // 近平面
    frustum.planes[5] = { -Front, Position + frontMultFar };     // 远平面

    return frustum;
}

bool Camera::CheckSphereVisibility(const vec3& center, float radius) const {
    const Frustum frustum = GetFrustum(Window::GetAspectRatio());
    
    for (int i = 0; i < 6; ++i) {
        const float distance = glm::dot(frustum.planes[i].normal, center) - glm::dot(frustum.planes[i].normal, frustum.planes[i].distance);

        if (distance < -radius) { return false; }
    }
    return true;
}

}