// include/managers/light.h
#pragma once
#include "managers/fwd.h"
#include "prefabs/volum_beam.h"

namespace CubeDemo::Managers {
class LightMng {

    inline static unsigned int m_InstCount = 0;

public:
    LightMng();
    static LightMng* CreateInst();

    // 定义光源加载结果结构体
    struct LightLoadResult {
        std::vector<DL*> dirLights;
        std::vector<PL*> pointLights;
        std::vector<SL*> spotLights;
        std::vector<SkL*> skyLights;
        std::vector<VolumBeam*> volumBeams;
    };

    // 灯光配置加载接口（声明）
    static LightLoadResult LoadLightConfigs(const string& config_path);
};

} // namespace CubeDemo::Managers
