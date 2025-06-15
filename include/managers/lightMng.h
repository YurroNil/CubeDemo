// include/managers/lightMng.h
#pragma once
#include "managers/fwd.h"
#include "managers/light/creater.h"
#include "managers/light/cleanner.h"

namespace CubeDemo::Managers {
class LightMng {

    inline static unsigned int s_InstCount = 0;

public:
    LightGetter Get; LightCreater Create; LightCleanner Remove;

    static LightMng* CreateInst();

    // 读取json配置文件的参数，并应用到光源的成员中
    template<typename T, typename... Args>
    static void SetLightsData(const string& config_path, T* first, Args*... args);
};
} // namespace CubeDemo::Managers

// 模板实现
#include "managers/light/utils.inl"
