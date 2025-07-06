// include/managers/model/mng.h
#pragma once
#include "resources/model.h"
#include "prefabs/lights/data.h"

namespace CubeDemo::Managers {

class ModelMng {
public:
    static ModelMng* CreateInst();
    static void RemoveInst(ModelMng** ptr);
    void AllUseShader(
        Camera* camera, float aspect_ratio,
        DL* dir_light, SL* spot_light,
        PL* point_light, SkL* sky_light
    );
    void RmvAllModels();
    void RmvAllShaders();

};
}   // namespace CubeDemo::Managers

// 全局别名
using ModelMng = CubeDemo::Managers::ModelMng;
