// src/core/camera.cpp

#include "core/camera.h"

Camera::Camera(vec3 position, vec3 up, float yaw, float pitch)
    : Position(position),
      WorldUp(up),
      Yaw(yaw),
      Pitch(pitch),
      Front(vec3(0.0f, 0.0f, -1.0f)), // 默认前向量
      MovementSpeed(2.5f),
      MouseSensitivity(0.1f),
      Zoom(45.0f)
{
    // 根据参数计算实际方向
    updateCameraVectors();
}

// 获取视图矩阵
mat4 Camera::GetViewMatrix() const {
    return glm::lookAt(Position, Position+Front, Up);
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
    
    default:break;
    }
}

// 处理鼠标移动
void Camera::ProcessMouseMovement(float xoffset, float yoffset) {

    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw += xoffset;
    Pitch += yoffset;

    // 限制俯仰角
    if (Pitch > 89.0f) { Pitch = 89.0f; }
    if (Pitch < -89.0f) { Pitch = -89.0f; }

    updateCameraVectors();
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


void Camera::updateCameraVectors() {
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
