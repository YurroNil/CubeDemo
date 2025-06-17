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
    Camera* camera, float aspect_ratio, // 相机, 纵横比
    DL* dir_light, SL* spot_light,      // 方向光, 聚光
    PL* point_light)                    // 点光
{
    // 链接着色器程序ID
    for(auto* model : MODEL_POINTERS) {

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

    }
}

void ModelMng::RmvAllModels() {
    for(auto* model : MODEL_POINTERS) {
        delete model;
    }
    MODEL_POINTERS.clear();
}

void ModelMng::RmvAllShaders() {
    for(auto* model : MODEL_POINTERS) {
        delete model->ModelShader; model->ModelShader = nullptr;
    }
}

ModelMng* ModelMng::CreateInst() {
    return new ModelMng();
}
void ModelMng::RemoveInst(ModelMng** ptr) {
    delete *ptr; *ptr = nullptr;
}

}
