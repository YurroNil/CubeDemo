// src/managers/light/utils.cpp
#include "pch.h"
#include "managers/light/mng.h"
#include "managers/light/utils.inl"

namespace CubeDemo::Managers {

LightMng::LightLoadResult LightMng::LoadLightConfigs(const string& config_path) {
    LightLoadResult result;
    json config = ReadConfig(config_path);
    
    if (config.is_array()) {
        for (const auto& item : config) {
            std::string type = item.value("type", "");
            
            if (type == "prefab.dir_light") {
                auto light_ptr = new DL();
                SetLightDataImpl(item, light_ptr);
                result.dirLights.push_back(light_ptr);
            }
            else if (type == "prefab.point_light") {
                auto light_ptr = new PL();
                SetLightDataImpl(item, light_ptr);
                result.pointLights.push_back(light_ptr);
            }
            else if (type == "prefab.spot_light") {
                auto light_ptr = new SL();
                SetLightDataImpl(item, light_ptr);
                result.spotLights.push_back(light_ptr);
            }
            else if (type == "prefab.sky_light") {
                auto light_ptr = new SkL();
                SetLightDataImpl(item, light_ptr);
                result.skyLights.push_back(light_ptr);
            }
            else if (type == "effect.volumetric") {
                auto beam_ptr = new VolumBeam();
                SetLightDataImpl(item, beam_ptr);
                result.volumBeams.push_back(beam_ptr);
            }
            else if (!type.empty()) {
                std::cerr << "警告：未处理的光源类型: " << type << std::endl;
            }
        }
    }
    else {
        std::cerr << "错误：无效的灯光配置文件格式" << std::endl;
    }
    
    return result;
}

} // namespace CubeDemo::Managers
