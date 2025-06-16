// include/resources/model.h
#pragma once
#include "graphics/bound_sphere.h"
#include "managers/model/getter.h"

namespace CubeDemo {

class Model : public Managers::ModelGetter {
    friend class Managers::ModelGetter;
public:
    // 创建包维球实例
    BoundingSphere bounds;

    Model(const string& path);  // 初始化
    void NormalDraw(Shader* shader, const vec3& camera_pos);
    void DrawCall(Shader* shader, Camera* camera);
    void DrawSimple() const; 

private:
    // 模型的基本属性 (不可见)

    // 基本要素
    string m_ID, m_Name, m_Type, m_Path, m_vshPath, m_fshPath;

    MeshArray m_Meshes;              // 网格数据
    mat4 m_ModelMatrix{mat4(1.0f)};  // 模型矩阵(用于控制模型在世界空间的变换，如偏移,旋转,缩放)
    vec3 m_Position{vec3(0.0f)};     // 位置
    float m_Rotation{0.0f};          // 旋转(弧度)
    vec3 m_Scale{vec3(1.0f)};        // 尺寸


    // 异步加载状态
    std::atomic<bool> m_isLoading = false;
    std::atomic<bool> m_MeshesReady = false;

    // 私有方法
    const void UpdateModelMatrix();

};
}   // namespace CubeDemo
