// include/resources/model.h
#pragma once
#include "graphics/bound_sphere.h"
#include "managers/model/getter.h"
#include "ui/fwd.h"

namespace CubeDemo {

class Model : public Managers::ModelGetter {
    friend class Managers::ModelGetter;

    friend class UI::ModelTablePanel;

public:
    // 创建包维球实例
    BoundingSphere bounds;
    Shader* ModelShader = nullptr;   // 模型着色器(可公共修改)

    Model(const string& path);
    ~Model();
    void Init();
    void NormalDraw(bool is_mainloop_draw);
    void DrawCall(Camera* camera, bool is_mainloop_draw = true);
    void Delete();
    void UseShaders(
        Camera* camera,
        DL* dir_light = nullptr, SL* spot_light = nullptr,
        PL* point_light = nullptr, SkL* sky_light = nullptr
    );

private:
    // 模型的基本属性 (不可见)

// ------------- 模型的基本属性 -------------

    // ID, 模型名, 类型, 路径等
    string
        m_ID, m_Name, m_Type, m_Path, m_IconPath,
        m_vshPath, m_fshPath, m_Description;

// ------------- 模型资源 -------------

    MeshArray m_Meshes;  // 网格数据

    // 模型矩阵(用于控制模型在世界空间的变换，如偏移,旋转,缩放)
    mat4 m_ModelMatrix{mat4(1.0f)};

    // 位置，旋转(弧度)，尺寸        // 数据备份, 用于撤销/重置
    vec3 m_Position = vec3(0.0f),   m_PosCopy = vec3(0.0f);
    vec3 m_Rotation = vec3(0.0f),   m_RotCopy = vec3(0.0f);
    vec3 m_Scale = vec3(1.0f),      m_ScaleCopy = vec3(1.0f);


// ------------- 加载状态 -------------
    std::atomic<bool> m_isLoading = false, m_MeshesReady = false;


// ------------- 私有方法 -------------
    const void UpdateModelMatrix();

};
}   // namespace CubeDemo
