// src/core/camera.cpp
#include "pch.h"
#include "core/camera.h"
#include "core/window.h"

namespace CubeDemo {
using namespace glm;

Camera::Camera(vec3 pos, vec3 up, float yaw, float pitch)
: Position(pos) {
    direction.front = vec3(0.0f, 0.0f, -1.0f);
    direction.worldUp = up; rotation = {yaw, pitch};
    attribute = {4.0f, 0.1f, 45.0f};
    UpdateCameraVec();
}

// 获取视图矩阵
mat4 Camera::GetViewMat() const {
    return lookAt(Position, Position+direction.front, direction.up);
}
// 获取投影矩阵
mat4 Camera::GetProjectionMat(float aspect) const {
    return perspective(radians(attribute.zoom), aspect, frustumPlane.near_plane, frustumPlane.far_plane);
}

// 处理鼠标移动
void Camera::ProcMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) {

    xoffset *= attribute.mouseSensvty; yoffset *= attribute.mouseSensvty;
    rotation.yaw += xoffset; rotation.pitch += yoffset;

    // 限制俯仰角
    if (rotation.pitch > 89.0f) { rotation.pitch = 89.0f; }
    if (rotation.pitch < -89.0f) { rotation.pitch = -89.0f; }

    UpdateCameraVec();
}

// 处理滚轮
void Camera::ProcMouseScroll(float yoffset) {
    attribute.zoom -= yoffset;
    if (attribute.zoom < 1.0f) attribute.zoom = 1.0f;
    if (attribute.zoom > 45.0f) attribute.zoom = 45.0f;
}

//跳跃(未时装)
void Camera::Jump(float velocity) { Position.y += velocity; }

void Camera::UpdateCameraVec() {
    // 根据欧拉角计算前向量
    vec3 front;
    front.x = cos(radians(rotation.yaw)) * cos(radians(rotation.pitch));
    front.y = sin(radians(rotation.pitch));
    front.z = sin(radians(rotation.yaw)) * cos(radians(rotation.pitch));
    direction.front = normalize(front);

    // 重新计算右向量和上向量
    direction.right = normalize(cross(direction.front, direction.worldUp));
    direction.up = normalize(cross(direction.right, direction.front));
}

void Camera::SaveCamera(Camera* c) { m_SaveCameraPtr = c; }

Camera* Camera::GetCamera() { return m_SaveCameraPtr; }

void Camera::Delete(Camera* c) { delete c; m_SaveCameraPtr = nullptr; }

 // 视锥体相关方法
Camera::Frustum Camera::GetFrustum(float aspectRatio) const {

    Frustum frustum;
    const float half_v_side = frustumPlane.far_plane * tanf(radians(attribute.zoom) * 0.5f);
    const float half_h_side = half_v_side * aspectRatio;
    const vec3 front_mult_far = frustumPlane.far_plane * direction.front;

    frustum.planes[0] = { normalize(cross(direction.up, front_mult_far + direction.right * half_h_side)), Position }; // 左平面
    frustum.planes[1] = { normalize(cross(front_mult_far - direction.right * half_h_side, direction.up)), Position }; // 右平面
    frustum.planes[2] = { normalize(cross(direction.right, front_mult_far - direction.up * half_v_side)), Position };  // 下平面
    frustum.planes[3] = { normalize(cross(front_mult_far + direction.up * half_v_side, direction.right)), Position };  // 上平面
    frustum.planes[4] = { direction.front, Position + direction.front * frustumPlane.near_plane }; // 近平面
    frustum.planes[5] = { -direction.front, Position + front_mult_far };     // 远平面

    return frustum;
}
// 检查视椎体是否可见
bool Camera::isSphereVisible(const vec3& center, float radius) const {
    const Frustum frustum = GetFrustum(Window::GetAspectRatio());
    
    for (int i = 0; i < 6; i++) {
        const float distance = dot(frustum.planes[i].normal, center) - dot(frustum.planes[i].normal, frustum.planes[i].distance);

        if (distance < -radius) { return false; }
    }
    return true;
}

vec3 Camera::ScreenToWorld(
    float screenX, float screenY,
    Camera* camera, float depth,
    int width, int height)
{
    // 将屏幕坐标转换为归一化设备坐标(NDC)
    float ndcX = (2.0f * screenX) / width - 1.0f;
    float ndcY = 1.0f - (2.0f * screenY) / height; // Y轴翻转
    
    // 创建NDC坐标点 (z值范围[-1,1]，depth参数范围[0,1])
    float ndcZ = 2.0f * depth - 1.0f;
    vec4 ndcPoint(ndcX, ndcY, ndcZ, 1.0f);
    
    // 计算视图和投影矩阵的逆矩阵
    float aspect = static_cast<float>(width) / height;
    mat4 view_matrix = camera->GetViewMat();
    mat4 proj_matrix = camera->GetProjectionMat(aspect);
    mat4 inv_view_proj = glm::inverse(proj_matrix * view_matrix);
    
    // 将NDC坐标转换为世界坐标
    vec4 world_Point = inv_view_proj * ndcPoint;
    
    // 透视除法 (将齐次坐标转换为3D坐标)
    if (world_Point.w != 0.0f) {
        world_Point /= world_Point.w;
    }
    
    return vec3(world_Point);
}

}