// include/managers/lightMng.h
#pragma once
#include "managers/fwd.h"
#include "managers/light/creater.h"

namespace CubeDemo::Managers {
class LightMng {

    inline static unsigned int s_InstCount = 0;

public:
    LightMng();

    LightCreater Create;

    static LightMng* CreateInst();

    // 读取json配置文件的参数，并应用到光源的成员中
    template<typename T, typename... Args>
    static void SetLightsData(const string& config_path, T* first, Args*... args);
};
} // namespace CubeDemo::Managers

// 模板实现
#include "managers/light/utils.inl"
