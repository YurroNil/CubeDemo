// include/core/camera.h

#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
using vec3 = glm::vec3;
using mat4 = glm::mat4;


class Camera {
public:
    // 相机属性
    vec3 Position;
    vec3 Front;
    vec3 Up;
    vec3 Right;
    vec3 WorldUp;

    // 欧拉角
    float Yaw;
    float Pitch;

    // 相机选项
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    Camera(
        vec3 position, 
        vec3 up,
        float yaw, 
        float pitch
    );

    mat4 GetViewMatrix() const;
    void ProcessKeyboard(int direction, float deltaTime);
    void ProcessMouseMovement(float xoffset, float yoffset);
    void ProcessMouseScroll(float yoffset);
    void Jump(float velocity);

private:
    void updateCameraVectors();
};
