// src/graphics/modelManager.cpp
#include "resources/modelManager.h"


void ModelManager::Register(const string& name, 
    const string& modelPath,
    const string& vshPath,
    const string& fshPath) {
    try {
        ModelInstance instance;
        instance.data = ModelLoader::LoadFromJson(modelPath);
        instance.shader = new Shader(vshPath, fshPath);
        instance.mesh = new Mesh(*instance.data);
        s_Models[name] = instance;
    } catch (const std::exception& e) {
        std::cerr << "Model Registration Failed: " << e.what() << std::endl;
    }
}
// 渲染循环
void ModelManager::Render(const string& name, const Camera& camera, float aspectRatio) {
    if (s_Models.find(name) != s_Models.end()) {
        auto& instance = s_Models[name];
        
        // 矩阵设置保持不变
        mat4 projection = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 100.0f);
        mat4 view = camera.GetViewMatrix();
        // 获取模型变换矩阵
        mat4 model = GetModelMatrix(name);
        instance.shader->Use();
        instance.shader->SetMat4("projection", projection);
        instance.shader->SetMat4("view", view);
        instance.shader->SetMat4("model", model);

        // 新增光照参数设置
        // 设置光照属性
        instance.shader->SetVec3("light.position", vec3(-0.5f, 2.0f, -2.0f));
        instance.shader->SetVec3("light.color", vec3(1.0f));
        instance.shader->SetFloat("light.ambientStrength", 0.1f);
        instance.shader->SetFloat("light.specularStrength", 0.5f);
        instance.shader->SetFloat("light.constant", 1.0f);
        instance.shader->SetFloat("light.linear", 0.09f);
        instance.shader->SetFloat("light.quadratic", 0.032f);
        // 设置材质属性
        instance.shader->SetVec3("material.ambient", instance.data->material.ambient);
        instance.shader->SetVec3("material.diffuse", instance.data->material.diffuse);
        instance.shader->SetVec3("material.specular", instance.data->material.specular);
        instance.shader->SetFloat("material.shininess", instance.data->material.shininess);
        // 设置相机位置
        instance.shader->SetVec3("viewPos", camera.Position);
        instance.mesh->Draw();
    }
}

// 删除模型
void ModelManager::Delete(const string& name) {
    if (s_Models.find(name) != s_Models.end()) {
        auto& instance = s_Models[name];
        delete instance.data;
        delete instance.shader;
        delete instance.mesh;
        s_Models.erase(name);
    }
}

void ModelManager::SetPosition(const string& name, const vec3& position) {
    if(s_Models.find(name) != s_Models.end()) {
        s_Models[name].transform.position = position;
    }
}

void ModelManager::SetRotation(const string& name, const vec3& rotation) {
    if(s_Models.find(name) != s_Models.end()) {
        s_Models[name].transform.rotation = rotation;
    }
}

void ModelManager::SetScale(const string& name, const vec3& scale) {
    if(s_Models.find(name) != s_Models.end()) {
        s_Models[name].transform.scale = scale;
    }
}

void ModelManager::Move(const string& name, const vec3& delta) {
    if(s_Models.find(name) != s_Models.end()) {
        s_Models[name].transform.position += delta;
    }
}

void ModelManager::Rotate(const string& name, const vec3& delta) {
    if(s_Models.find(name) != s_Models.end()) {
        s_Models[name].transform.rotation += delta;
    }
}

void ModelManager::Scale(const string& name, const vec3& delta) {
    if(s_Models.find(name) != s_Models.end()) {
        s_Models[name].transform.scale += delta;
    }
}

mat4 ModelManager::GetModelMatrix(const string& name) {
    if(s_Models.find(name) != s_Models.end()) {
        auto& t = s_Models[name].transform;
        mat4 model = mat4(1.0f);
        model = glm::translate(model, t.position);
        model = glm::rotate(model, glm::radians(t.rotation.x), vec3(1,0,0));
        model = glm::rotate(model, glm::radians(t.rotation.y), vec3(0,1,0));
        model = glm::rotate(model, glm::radians(t.rotation.z), vec3(0,0,1));
        model = glm::scale(model, t.scale);
        return model;
    }
    return mat4(1.0f);
}