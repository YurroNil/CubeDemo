// include/loaders/modelIniter.h
#pragma once
#include "utils/defines.h"
#include "utils/jsonConfig.h"
#include "resources/model.h"

namespace CubeDemo {

class Loaders::ModelIniter {
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

    // 模型配置
    static constexpr const char* SHADER_VERTEX = VSH_PATH "model.glsl";
    static constexpr const char* SHADER_FRAGMENT = FSH_PATH "model.glsl";
};
}