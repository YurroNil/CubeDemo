#pragma once
#include <3rd-lib/glm/glm.hpp>
#include <3rd-lib/glm/gtc/matrix_transform.hpp>

class Camera {
public:
    // 相机属性
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    // 欧拉角
    float Yaw;
    float Pitch;

    // 相机选项
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    Camera(
        glm::vec3 position, 
        glm::vec3 up,
        float yaw, 
        float pitch
    );

    glm::mat4 GetViewMatrix() const;

    void ProcessKeyboard(int direction, float deltaTime);

    void ProcessMouseMovement(float xoffset, float yoffset);

    void ProcessMouseScroll(float yoffset);

    void Jump(float velocity);

private:
    void updateCameraVectors();
};
