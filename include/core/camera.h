// include/core/camera.h
#pragma once

namespace CubeDemo {

class Camera {
public:
    Camera(vec3 pos, vec3 up, float yaw, float pitch);
    ~Camera() = default;

    vec3 Position;

/* ------------Structs------------ */

    // Euler Angle
    struct Rotation { float yaw, pitch; } rotation;

    struct Direction { vec3 front, up, right, worldUp; } direction;

    struct Attributes {
        float movementSpeed = 2.0, // 移动速度
        mouseSensvty,        // Mouse Senstivity的简写, 即鼠标灵敏度
        zoom;                // 缩放
    } attribute;

    struct FrustumPlane {
        vec3 normal, distance;
        float near_plane{0.1f}, far_plane{100.0f};
    } frustumPlane;
    // 顺序：左、右、下、上、近、远
    struct Frustum { FrustumPlane planes[6]; };

/* ------------Getters------------ */
    mat4 GetViewMat() const;
    mat4 GetProjectionMat(float aspect) const;
    static Camera* GetCamera() { return m_SaveCameraPtr; }
    Frustum GetFrustum(float aspect) const;
    bool isSphereVsble(const vec3& center, float radius) const;

/* ----Proceesors and Actions----- */

    void ProcMouseMovement(float xoffset, float yoffset, bool constrainPitch);
    void ProcMouseScroll(float yoffset);
    void Jump(float velocity);
    
    static void SaveCamera(Camera* c);  // 保存摄像机对象的指针
    static void Delete(Camera* c);  // Cleanner
    
    static vec3 ScreenToWorld(
        float screenX, float screenY,
        Camera* camera, float depth,
        int viewport_width, int viewport_height
    );

    void TeleportTo(const vec3& position, float height_offset);

private:
    float m_lastX, m_lastY;
    void UpdateCameraVec();
    inline static Camera* m_SaveCameraPtr = nullptr;
};
}
