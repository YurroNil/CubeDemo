// src/managers/light.cpp
#include "pch.h"
#include "managers/light.h"
#include "managers/light_utils.inl"

// 别名
using Shader = CubeDemo::Shader;
using VolumBeam = CubeDemo::Prefabs::VolumBeam;
using BeamEffects = CubeDemo::Prefabs::BeamEffects;
namespace LightMapper = CubeDemo::Prefabs::Lights::Mapper;

namespace CubeDemo {
    namespace Managers {
        LightMng::LightMng() {
        }

        // 创建场景管理器
        LightMng *LightMng::CreateInst() {
            if (m_InstCount > 0) {
                std::cerr << "[LightMng] 光源创建失败，因为当前光源管理器数量为: " << m_InstCount << std::endl;
                return nullptr;
            }
            m_InstCount++;
            return new LightMng();
        }

        LightMng::LightLoadResult LightMng::LoadLightConfigs(const string &config_path) {
            LightLoadResult result;
            json config = ReadConfig(config_path);

            if (config.is_array()) {
                for (const auto &item: config) {
                    string type = item.value("type", "");

                    if (type == "prefab.dir_light") {
                        auto light_ptr = new DL();
                        SetLightDataImpl(item, light_ptr);
                        result.dirLights.push_back(light_ptr);
                    } else if (type == "prefab.point_light") {
                        auto light_ptr = new PL();
                        SetLightDataImpl(item, light_ptr);
                        result.pointLights.push_back(light_ptr);
                    } else if (type == "prefab.spot_light") {
                        auto light_ptr = new SL();
                        SetLightDataImpl(item, light_ptr);
                        result.spotLights.push_back(light_ptr);
                    } else if (type == "prefab.sky_light") {
                        auto light_ptr = new SkL();
                        SetLightDataImpl(item, light_ptr);
                        result.skyLights.push_back(light_ptr);
                    } else if (type == "effect.volumetric") {
                        auto beam_ptr = new VolumBeam();
                        SetLightDataImpl(item, beam_ptr);
                        result.volumBeams.push_back(beam_ptr);
                    } else if (!type.empty()) {
                        std::cerr << "警告：未处理的光源类型: " << type << std::endl;
                    }
                }
            } else {
                std::cerr << "错误：无效的灯光配置文件格式" << std::endl;
            }

            return result;
        }
    } // namespace Managers

    namespace Prefabs::Lights::Mapper {
        // 显式特化的实现
        template<>
        void MapLightData<BeamEffects::FlickerParams>(const json &j, BeamEffects::FlickerParams &flicker) {
            if (j.contains("enable")) flicker.enable = j["enable"].get<bool>();
            if (j.contains("min")) flicker.min = j["min"].get<float>();
            if (j.contains("max")) flicker.max = j["max"].get<float>();
            if (j.contains("speed")) flicker.speed = j["speed"].get<float>();
        }

        // 显式实例化需要的模板
        template void MapLightData<DL>(const json &, DL &);

        template void MapLightData<PL>(const json &, PL &);

        template void MapLightData<SL>(const json &, SL &);

        template void MapLightData<SkL>(const json &, SkL &);

        template void MapLightData<BeamEffects>(const json &, BeamEffects &);
    } // namespace Prefabs::Lights::Mapper
} // namespace CubeDemo
