// src/managers/model/getter.cpp
#include "pch.h"
#include "managers/model/getter.h"
#include "resources/model.h"

using AtomBool = std::atomic<bool>;

namespace CubeDemo::Managers {

// 构造函数
ModelGetter::ModelGetter(::CubeDemo::Model* model)
    : m_owner(model) {}

/* ----------- Setters ----------- */
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
const void ModelGetter::SetRotation(float rotation) {
    m_owner->m_Rotation = rotation;
    m_owner->UpdateModelMatrix();
}
const void ModelGetter::SetScale(vec3 scale) {
    m_owner->m_Scale = scale;
    m_owner->UpdateModelMatrix();
}
const void ModelGetter::SetTransform(const vec3& pos, float rotation, const vec3& scale) {
    m_owner->m_Position = pos;
    m_owner->m_Rotation = rotation;
    m_owner->m_Scale = scale;
    m_owner->UpdateModelMatrix();
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
const float ModelGetter::GetRotation() const {
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
}   // namespace CubeDemo::Managers
