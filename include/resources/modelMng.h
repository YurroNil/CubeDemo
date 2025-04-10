// include/resources/modelMng.h

#pragma once

#include "resources/modelLoader.h"
#include "graphics/shader.h"
#include "graphics/mesh.h"
namespace CubeDemo {

class ModelMng {
public:
    struct Transform {
        vec3 position = vec3(0.0f); // 模型的位置坐标
        vec3 rotation = vec3(0.0f); // 模型的旋转(欧拉角, 单位：度)
        vec3 scale = vec3(1.0f);    // 模型的尺寸
    };

    struct ModelInstance {
        ModelData* data; Shader* shader; Mesh* mesh; Transform transform; };

    static void Register(
        const string& name,         // 模型名
        const string& modelPath,    // 模型元数据文件路径
        const string& vshPath,      // 顶点着色器路径
        const string& fshPath       // 片段着色器路径
        );
    
    // 渲染(在渲染循环中渲染指定的模型)
    static void Render(const string& name, const Camera& camera, float aspectRatio);
    static void Delete(const string& name);



    // 变换控制方法
    static void SetPosition(const string& name, const vec3& position);
    static void SetRotation(const string& name, const vec3& rotation);
    static void SetScale(const string& name, const vec3& scale);
    static void Move(const string& name, const vec3& delta);
    static void Rotate(const string& name, const vec3& delta);
    static void Scale(const string& name, const vec3& delta);
    static mat4 GetModelMatrix(const string& name); // 获取变换矩阵


private:
    inline static std::unordered_map<string, ModelInstance> s_Models;
};

}