// src/managers/modelMng.cpp
#include "pch.h"
#include "managers/modelMng.h"
#include "graphics/shader.h"

namespace CubeDemo {
    extern ModelMng* MODEL_MNG;
    extern std::vector<Model*> MODEL_POINTERS;
}

namespace CubeDemo::Managers {

void ModelMng::AllUseShader(
    Camera* camera, float aspect_ratio,   // 相机, 纵横比
    DL* dir_light, SL* spot_light,        // 方向光, 聚光
    PL* point_light, SkL* sky_light)      // 点光
{
    if(MODEL_POINTERS.empty()) return;
    // 链接着色器程序ID
    for(auto* model : MODEL_POINTERS) {
        if(model == nullptr) continue;
        // 使用着色器
        model->ModelShader->Use();
        // 摄像机参数传递
        model->ModelShader->ApplyCamera(camera, aspect_ratio);
        // 设置位置
        model->ModelShader->SetViewPos(camera->Position);

        // 传递模型的着色器指针，给光源setter来获取光源信息
        if(dir_light != nullptr) dir_light->SetShader(*model->ModelShader);
        if(spot_light != nullptr) spot_light->SetShader(*model->ModelShader);
        if(point_light != nullptr) point_light->SetShader(*model->ModelShader);
        if(sky_light != nullptr) sky_light->SetShader(*model->ModelShader);

    }
}

// 删除所有着色器以及模型
void ModelMng::RmvAllModels() {
    if(MODEL_POINTERS.empty()) return;
    
    // 删除所有模型
    for(auto* model : MODEL_POINTERS) {
        if(model == nullptr) continue;
            std::cout << "  删除模型: " << model << " (" << model->GetID() << ")" << std::endl;
            delete model;
    }
    MODEL_POINTERS.clear();
}

// 删除所有着色器(暂不使用)
void ModelMng::RmvAllShaders() {
    for(auto* model : MODEL_POINTERS) {
        if (model == nullptr || model->ModelShader == nullptr) continue;
        delete model->ModelShader;
        model->ModelShader = nullptr;
    }
}
// 创建模型管理器实例
ModelMng* ModelMng::CreateInst() {
    return new ModelMng();
}
// 移除模型管理器实例
void ModelMng::RemoveInst(ModelMng** ptr) {
    delete *ptr; *ptr = nullptr;
}
}   // namespace CubeDemo::Managers
