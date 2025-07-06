// src/managers/light/mng.cpp
#include "pch.h"
#include "managers/light/mng.h"
#include "prefabs/lights/volum_beam.h"

// 别名
using Shader = CubeDemo::Shader;
using VolumBeam = CubeDemo::Prefabs::VolumBeam;

namespace CubeDemo {
namespace Managers {

LightMng::LightMng() {
    // 显式初始化成员
    Create = LightCreater();
}

// 创建场景管理器
LightMng* LightMng::CreateInst() {
    if(m_InstCount > 0) {
        std::cerr << "[LightMng] 光源创建失败，因为当前光源管理器数量为: " << m_InstCount << std::endl;
        return nullptr;
    }
    m_InstCount++;
    return new LightMng();
}}}   // namespace
