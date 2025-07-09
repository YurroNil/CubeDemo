// src/scenes/dynamic_scene.cpp
#include "pch.h"
#include "scenes/dynamic_scene.h"
#include "resources/model.h"
#include "managers/model/mng.h"
#include "loaders/texture.h"

namespace CubeDemo {
    extern Managers::ModelMng* MODEL_MNG;
    extern std::vector<Model*> MODEL_POINTERS;
}

namespace CubeDemo::Scenes {

DynamicScene::DynamicScene(const SceneInfo& info) 
    : m_info(info), SceneBase(info) {
}
void DynamicScene::Init() {
    // 初始化场景中的模型
    for (auto* model : MODEL_POINTERS) {
        model->Init();
    }
    // 初始化光源
    for (auto* light : m_dirLights) {
        light->Init();
    }
    for (auto* light : m_pointLights) {
        light->Init();
    }
    for (auto* light : m_spotLights) {
        light->Init();
    }
    for (auto* light : m_skyLights) {
        light->Init();
    }
    for (auto* beam : m_volumBeams) {
        beam->Init();
    }
}

void DynamicScene::Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) {

    // 视口设置
    glViewport(0, 0, WINDOW::GetWidth(), WINDOW::GetHeight());
    
    // 清除缓冲区
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // 获取第一个光源（如果有的话）用于设置着色器
    DL* dirLight = m_dirLights.empty() ? nullptr : m_dirLights[0];
    PL* pointLight = m_pointLights.empty() ? nullptr : m_pointLights[0];
    SL* spotLight = m_spotLights.empty() ? nullptr : m_spotLights[0];
    SkL* skyLight = m_skyLights.empty() ? nullptr : m_skyLights[0];
    
    // 更新所有模型的着色器uniform
    MODEL_MNG->AllUseShader(
        camera, WINDOW::GetAspectRatio(),
        dirLight, spotLight,
        pointLight, skyLight
    );

    // 渲染所有模型
    if (!MODEL_POINTERS.empty()) {
        for (auto* model : MODEL_POINTERS) {
            model->DrawCall(camera);
        }
    }

    if(spotLight == nullptr || m_volumBeams.empty()) return;
    // 渲染体积光
    for (auto* beam : m_volumBeams) {
        if(beam == nullptr || beam->VolumShader == nullptr) continue;
        beam->Render(camera, spotLight);
    }
}

void DynamicScene::Cleanup() {
    // 清理模型 - 确保正确释放模型资源
    MODEL_MNG->RmvAllModels();

    // 清理光源
    for (auto* light : m_dirLights) {
        if (light) {
            delete light;
        }
    }
    m_dirLights.clear();
    
    for (auto* light : m_pointLights) {
        if (light) {
            delete light;
        }
    }
    m_pointLights.clear();
    
    for (auto* light : m_spotLights) {
        if (light) {
            delete light;
        }
    }
    m_spotLights.clear();
    
    for (auto* light : m_skyLights) {
        if (light) {
            delete light;
        }
    }
    m_skyLights.clear();
    
    // 清理体积光
    for (auto* beam : m_volumBeams) {
        if (beam) {
            delete beam;
        }
    }
    m_volumBeams.clear();
    
    // 清理阴影贴图
    for (auto* shadowMap : m_shadowMaps) {
        if (shadowMap) {
            // delete shadowMap;
        }
    }
    m_shadowMaps.clear();
    
    // 额外清理：确保纹理缓存也被清理
    TL::ClearCache();
}

// 添加模型
void DynamicScene::AddModel(::CubeDemo::Model* model) { MODEL_POINTERS.push_back(model); }
// 添加方向光
void DynamicScene::AddDirLight(Prefabs::DirLight* light) { m_dirLights.push_back(light); }
// 添加点光源
void DynamicScene::AddPointLight(Prefabs::PointLight* light) { m_pointLights.push_back(light); }
// 添加聚光灯
void DynamicScene::AddSpotLight(Prefabs::SpotLight* light) { m_spotLights.push_back(light); }
// 添加天空光
void DynamicScene::AddSkyLight(Prefabs::SkyLight* light) { m_skyLights.push_back(light); }
// 添加体积光
void DynamicScene::AddVolumBeam(Prefabs::VolumBeam* beam) { m_volumBeams.push_back(beam); }
// 添加阴影贴图
void DynamicScene::AddShadowMap(Prefabs::ShadowMap* shadowMap) { m_shadowMaps.push_back(shadowMap); }
} // namespace CubeDemo::Scenes
