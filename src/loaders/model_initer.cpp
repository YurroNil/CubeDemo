// src/loaders/model_initer.cpp
#include "pch.h"
#include "loaders/model_initer.h"
#include "threads/task_queue.h"
#include "graphics/shader.h"
#include "utils/defines.h"
#include "managers/sceneMng.h"

namespace CubeDemo {

// 别名
using MIL = Loaders::ModelIniter;
using UMC = Utils::JsonConfig;

// 外部变量声明
extern bool DEBUG_ASYNC_MODE;
extern std::vector<::CubeDemo::Model*> MODEL_POINTERS;
extern SceneMng* SCENE_MNG;

void MIL::InitModels() {
    std::cout << "\n[INITER] 模型初始化开始" << std::endl;
    
    try {
        // 根据场景情况进行初始化路径
        string curr_scene_name = SCENE_MNG->GetCurrentScene.ID();

        // "resources/scenes/" + 当前场景名 + "/model.json"
        string scene_config_path = SCENE_CONF_PATH + curr_scene_name + string("/models.json");

        // 加载模型列表
        const auto model_list = UMC::LoadModelConfig(scene_config_path);

        // 加载每个模型
        for (const auto& config : model_list) {
            // "resources/models/" + "模型名/模型名.obj" (或者其它模型格式，如fbx)
            const string full_path = string(MODEL_PATH) + config.path;
            LoadSingleModel(full_path, config);
        }

        std::cout << "[INITER] 成功加载 " << MODEL_POINTERS.size() << " 个模型" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] 模型初始化失败: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MIL::LoadSingleModel(const string& model_path, const Utils::ModelConfig& config) {
    try {
        // 创建模型实例
        ::CubeDemo::Model* model = new ::CubeDemo::Model(model_path);

        // 设置加载装填
        std::atomic<bool> model_loaded{false};

        // 创建模型加载器实例，并绑定当前函数创建的model指针
        ML model_loader(model_path, model);

        // 应用参数到模型数据上
        model->SetID(config.id);
        model->SetName(config.name);
        model->SetType(config.type);
        model->SetPosition(config.position);
        model->SetRotation(config.rotation);
        model->SetScale(config.scale);
        model->SetShaderPaths(config.vsh_path, config.fsh_path);

        // 创建模型着色器
        model->CreateShader();

        LoadModelData(model_loaded, &model_loader, DEBUG_ASYNC_MODE);
        WaitForModelLoad(model_loaded);
        
        // 储存创建好的model指针
        MODEL_POINTERS.push_back(model);
        ValidateModelData(model);
        
        std::cout << "[INITER] 已加载模型: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] 模型加载失败: " << model_path << " - " << e.what() << std::endl;
        
        throw; // 向上传递异常以终止初始化
    }
}

// 加载模型数据的核心逻辑
void MIL::LoadModelData(
    std::atomic<bool>& model_loaded,
    ML* model_loader,
    bool async_mode)
{
    if (async_mode) {
        model_loader->LoadAsync([&]{ model_loaded.store(true); });
    } else {
        model_loader->LoadSync([&]{ model_loaded.store(true); });
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

// 模型数据验证
void MIL::ValidateModelData(::CubeDemo::Model* model) {
    if (model->bounds.Rad < 0.01f) {
        std::cerr << "[WARNING] 模型包围球异常，可能未正确加载顶点数据" << std::endl;
        // 这里可以添加更多验证逻辑，比如检查顶点数量等
    }
}

} // namespace CubeDemo
