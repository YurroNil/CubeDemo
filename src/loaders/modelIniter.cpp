// src/loaders/modelIniter.cpp
#include "pch.h"
#include "loaders/modelIniter.h"
#include "utils/jsonConfig.h"
#include "threads/taskQueue.h"
#include "graphics/shader.h"
#include "utils/defines.h"

// 别名
using MIL = CubeDemo::Loaders::ModelIniter;

namespace CubeDemo {

// 外部变量声明
extern bool DEBUG_ASYNC_MODE;
extern std::vector<CubeDemo::Model*> MODEL_POINTERS;
extern Shader* MODEL_SHADER;

void MIL::InitModels() {
    std::cout << "\n[INITER] 模型初始化开始" << std::endl;
    
    try {
        // 加载模型列表
        const auto model_list = Utils::JsonConfig::LoadModelList("../resources/models/config.json");
        
        // 初始化着色器（所有模型共享）
        MODEL_SHADER = new Shader(
            VSH_PATH + string("model.glsl"),
            FSH_PATH + string("model.glsl")
        );

        // 加载每个模型
        for (const auto& model_rel_path : model_list) {
            const string full_path = string(MODEL_PATH) + model_rel_path;
            LoadSingleModel(full_path);
        }

        std::cout << "[INITER] 成功加载 " << MODEL_POINTERS.size() << " 个模型" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] 模型初始化失败: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MIL::LoadSingleModel(const string& model_path) {
    try {
        CubeDemo::Model* model = new CubeDemo::Model(model_path);
        std::atomic<bool> model_loaded{false};

        LoadModelData(model_loaded, model, DEBUG_ASYNC_MODE);
        WaitForModelLoad(model_loaded);
        
        MODEL_POINTERS.push_back(model);
        ValidateModelData(model);
        
        std::cout << "[INITER] 已加载模型: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] 模型加载失败: " << model_path << " - " << e.what() << std::endl;
        
        throw; // 向上传递异常以终止初始化
    }
}

// 加载模型数据的核心逻辑
void MIL::LoadModelData(std::atomic<bool>& model_loaded, CubeDemo::Model* model, bool async_mode) {
    if (async_mode) {
        model->LoadAsync([&]{ model_loaded.store(true); });
    } else {
        model->LoadSync([&]{ model_loaded.store(true); });
    }
}

// 监控加载状态的等待循环
void MIL::WaitForModelLoad(std::atomic<bool>& model_loaded) {
    auto start_time = csclock::now();
    
    while (!model_loaded.load()) {
        int processed = 0;
        TaskQueue::ProcTasks(processed);
        
        CheckForTimeout(start_time);  // 检查超时
        
        // 无任务时让出CPU
        if (processed == 0) {
            std::this_thread::yield();
        }
    }
}

// 超时检测
void MIL::CheckForTimeout(const std::chrono::time_point<csclock>& start_time) {
    constexpr auto timeout = std::chrono::seconds(3);
    if (csclock::now() > start_time + timeout) {
        throw std::runtime_error("模型加载超时");
    }
}

// 着色器初始化
void MIL::InitModelShader(const string& vsh_path, const string& fsh_path) {
    if (MODEL_SHADER != nullptr) {
        delete MODEL_SHADER;
    }
    MODEL_SHADER = new Shader(vsh_path, fsh_path);
}

// 模型数据验证
void MIL::ValidateModelData(CubeDemo::Model* model) {
    if (model->bounds.Rad < 0.01f) {
        std::cerr << "[WARNING] 模型包围球异常，可能未正确加载顶点数据" << std::endl;
        // 这里可以添加更多验证逻辑，比如检查顶点数量等
    }
}

} // namespace CubeDemo