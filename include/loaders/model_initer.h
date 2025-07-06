// include/loaders/model_initer.h
#pragma once

namespace CubeDemo {
class Model;
}

namespace CubeDemo::Loaders {
class Model;
using ML = ::CubeDemo::Loaders::Model;

class ModelIniter {
public:
    inline static bool s_isInitPhase = true;

    static void RemoveAllModels();
    static void InitModels();
    static void SwitchScene(const string& sceneID);
    
private:
    // 子功能模块
    static void LoadModelData(
        std::atomic<bool>& model_loaded,
        ML* model_loader,
        bool async_mode
    );

    static void WaitForModelLoad(std::atomic<bool>& model_loaded);
    static void CheckForTimeout(const std::chrono::time_point<csclock>& start_time);
    static void ValidateModelData(::CubeDemo::Model* model);
    static void LoadSingleModel(const string& model_path, const Utils::ModelConfig& config);
};
}   // namespace CubeDemo

using MIL = CubeDemo::Loaders::ModelIniter;
