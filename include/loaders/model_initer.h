// include/loaders/model_initer.h
#pragma once
#include "resources/model.h"

namespace CubeDemo {

class Loaders::ModelIniter {

    using csclock = std::chrono::steady_clock;
public:
    static void InitModels();
    
private:
    // 子功能模块
    static void LoadModelData(std::atomic<bool>& model_loaded, CubeDemo::Model* model, bool async_mode);
    static void WaitForModelLoad(std::atomic<bool>& model_loaded);
    static void CheckForTimeout(const std::chrono::time_point<csclock>& start_time);
    static void ValidateModelData(CubeDemo::Model* model);
    static void LoadSingleModel(const string& model_path);
    static void InitModelShader(const string& vsh_path, const string& fsh_path);
};
}