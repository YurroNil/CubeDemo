// include/managers/model/getter.h
#pragma once

namespace CubeDemo::Managers {
using MeshArray = std::vector<::CubeDemo::Mesh>;

class ModelGetter {
public:
    ModelGetter(::CubeDemo::Model* model);

    // Getters
    MeshArray& GetMeshes();
    const mat4& GetModelMatrix() const;
    bool IsReady() const;
    const std::atomic<bool>& isLoading() const;
    const vec3 GetPosition() const;
    const float GetRotation() const;
    const vec3 GetScale() const;
    const string GetID() const;
    const string GetName() const;
    const string GetType() const;
    const string GetVshPath() const;
    const string GetFshPath() const;

    // Setters
    const void InitModelAttri(const Utils::ModelConfig& config);

    std::atomic<bool>& SetMeshMarker();
    std::atomic<bool>& SetLoadingMarker();
    const void SetID(const string& id);
    const void SetName(const string& name);
    const void SetType(const string& type);
    const void SetPosition(vec3 pos);
    const void SetRotation(float rotation);
    const void SetScale(vec3 scale);
    const void SetTransform(const vec3& pos, float rotation, const vec3& scale);
    const void SetShaderPaths(const string& vsh_path, const string& fsh_path);
    const void CreateShader();
private:
    ::CubeDemo::Model* m_owner;
};
}