// include/loaders/model_initer.h
#pragma once
#include "loaders/model.h"
#include "resources/model.h"

using ML = CubeDemo::Loaders::Model;

namespace CubeDemo {

class Loaders::ModelIniter {

    using csclock = std::chrono::steady_clock;
public:
    static void InitModels();
    
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
}