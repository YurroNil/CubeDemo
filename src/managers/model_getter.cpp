// src/managers/model_getter.cpp
#include "pch.h"
#include "managers/model_getter.h"
#include "resources/model.h"
#include "utils/defines.h"

using AtomBool = std::atomic<bool>;

namespace CubeDemo::Managers {

// 构造函数
ModelGetter::ModelGetter(::CubeDemo::Model* model)
    : m_owner(model) {}

/* ----------- Setters ----------- */
const void ModelGetter::InitModelAttri(const Utils::ModelConfig& config) {
    // ID, 名字, 类型名, 路径
    m_owner->m_ID = config.id;
    m_owner->m_Name = config.name;
    m_owner->m_Type = config.type;
    m_owner->m_vshPath = VSH_PATH + config.vsh_path;
    m_owner->m_fshPath = FSH_PATH + config.fsh_path;
    m_owner->m_IconPath = MODEL_PATH + config.icon_path;
    m_owner->m_Description = config.description;

    // transform三要素(位置, 旋转, 缩放)
    m_owner->m_PosCopy = m_owner->m_Position = config.position;

    m_owner->m_RotCopy.x = m_owner->m_Rotation.x = config.rotation.x;
    m_owner->m_RotCopy.y = m_owner->m_Rotation.y = config.rotation.y;
    m_owner->m_RotCopy.z = m_owner->m_Rotation.z = config.rotation.z;
    
    m_owner->m_ScaleCopy = m_owner->m_Scale = config.scale;

    // 更新模型矩阵
    m_owner->UpdateModelMatrix();
}

const void ModelGetter::SetID(const string& id) {
    m_owner->m_ID = id;
}
const void ModelGetter::SetName(const string& name) {
    m_owner->m_Name = name;
}
const void ModelGetter::SetType(const string& type) {
    m_owner->m_Type = type;
}
AtomBool& ModelGetter::SetMeshMarker() {
    return m_owner->m_MeshesReady;
}
AtomBool& ModelGetter::SetLoadingMarker() {
    return m_owner->m_isLoading;
}
const void ModelGetter::SetPosition(vec3 pos) {
    m_owner->m_Position = pos;
    m_owner->UpdateModelMatrix();
}
const void ModelGetter::SetRotation(vec3 rotation) {
    m_owner->m_Rotation = rotation;
    m_owner->UpdateModelMatrix();
}
const void ModelGetter::SetScale(vec3 scale) {
    m_owner->m_Scale = scale;
    m_owner->UpdateModelMatrix();
}
const void ModelGetter::SetTransform(const vec3& pos, vec3 rotation, const vec3& scale) {
    m_owner->m_Position = pos;
    m_owner->m_Rotation = rotation;
    m_owner->m_Scale = scale;
    m_owner->UpdateModelMatrix();
}
const void ModelGetter::SetShaderPaths(const string& vsh_path, const string& fsh_path) {
    m_owner->m_vshPath = VSH_PATH + vsh_path;
    m_owner->m_fshPath = FSH_PATH + fsh_path;
}

/* ----------- Getters ----------- */
const string ModelGetter::GetID() const {
    return m_owner->m_ID;
}
const string ModelGetter::GetName() const {
    return m_owner->m_Name;
}
const string ModelGetter::GetType() const {
    return m_owner->m_Type;
}
const vec3 ModelGetter::GetPosition() const {
    return m_owner->m_Position;
}
const vec3 ModelGetter::GetRotation() const {
    return m_owner->m_Rotation;
}
const vec3 ModelGetter::GetScale() const {
    return m_owner->m_Scale;
}
MeshArray& ModelGetter::GetMeshes() {
    return m_owner->m_Meshes;
}
const mat4& ModelGetter::GetModelMatrix() const {
    return m_owner->m_ModelMatrix;
}
bool ModelGetter::IsReady() const {
    return m_owner->m_MeshesReady.load(std::memory_order_acquire);
}
const AtomBool& ModelGetter::isLoading() const {
    return m_owner->m_isLoading;
}
const string ModelGetter::GetVshPath() const {
    return m_owner->m_vshPath;
}
const string ModelGetter::GetFshPath() const {
    return m_owner->m_fshPath;
}
const void ModelGetter::CreateShader() {
    m_owner->ModelShader = new Shader(
        m_owner->m_vshPath,
        m_owner->m_fshPath
    );
}
}   // namespace CubeDemo::Managers
