// src/loaders/model_initer.cpp
#include "pch.h"
#include "loaders/MIL_inc.h"
#include "scenes/dynamic_scene.h"

namespace CubeDemo {
extern bool DEBUG_ASYNC_MODE;

// 别名
using UMC = Utils::JsonConfig;

// 外部变量声明
extern bool DEBUG_ASYNC_MODE; extern unsigned int DEBUG_INFO_LV;
extern std::vector<::CubeDemo::Model*> MODEL_POINTERS;
extern SceneMng* SCENE_MNG;

void MIL::InitModels() {
    // 检查场景管理器是否初始化
    if (!SCENE_MNG || !SCENE_MNG->GetCurrentScene()) {
        throw std::runtime_error("场景管理器未初始化");
    }
    
    // 获取当前场景
    auto* scene = dynamic_cast<Scenes::DynamicScene*>(SCENE_MNG->GetCurrentScene());
    if (!scene) {
        throw std::runtime_error("当前场景不是动态场景类型");
    }
    
    // 重置进度跟踪器
    ProgressTracker::Get().Reset();
    
    // 加载场景中的模型
    const auto& info = scene->GetSceneInfo();
    for (const auto& prefab : info.prefabs) {
        if (prefab.type != "model") continue;
        
        string fullPath = info.resourcePath + "/" + prefab.path;
        auto modelConfigs = Utils::JsonConfig::LoadModelConfig(fullPath);
        
        // 预注册所有模型资源
        for (const auto& config : modelConfigs) {
            const string full_path = MODEL_PATH + config.path;
            
            ProgressTracker::Get().AddResource(
                ProgressTracker::MODEL_FILE, 
                full_path
            );
            
            ProgressTracker::Get().AddResource(
                ProgressTracker::MODEL_GEOMETRY, 
                full_path
            );
        }
        // 加载每个模型
        for (const auto& config : modelConfigs) {
            const string full_path = MODEL_PATH + config.path;
            LoadSingleModel(full_path, config);
        }
    }

    // 加载场景中的光源
    for (const auto& prefab : info.prefabs) {
        if (prefab.type != "light") continue;
        
        string fullPath = info.resourcePath + "/" + prefab.path;

        // 使用光源管理器加载配置
        auto lightResult = LightMng::LoadLightConfigs(fullPath);
        
        // 将加载的光源添加到场景
        for (auto* light : lightResult.dirLights) {
            scene->AddDirLight(light);
        }
        for (auto* light : lightResult.pointLights) {
            scene->AddPointLight(light);
        }
        for (auto* light : lightResult.spotLights) {
            scene->AddSpotLight(light);
        }
        for (auto* light : lightResult.skyLights) {
            scene->AddSkyLight(light);
        }
        for (auto* beam : lightResult.volumBeams) {
            scene->AddVolumBeam(beam);
        }
    }
    // 模型加载完成后，更新面板信息
    UIMng::PanelUpdate();
    // 更新鼠标状态
    glfwSetInputMode(WINDOW::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    s_isInitPhase = false;
}

void MIL::SwitchScene(const string& sceneID) {
    // 卸载当前场景所有模型
    RemoveAllModels();
    
    // 切换到新场景
    SCENE_MNG->SwitchTo(sceneID);
    
    // 加载新场景模型
    InitModels();
}

void MIL::RemoveAllModels() {
    // 清理全局模型列表
    for (auto* model : MODEL_POINTERS) {
        delete model;
    }
    MODEL_POINTERS.clear();
    
    // 清理场景中的模型引用
    if (auto* scene = dynamic_cast<Scenes::DynamicScene*>(SCENE_MNG->GetCurrentScene())) {
        scene->Cleanup();
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
        model->InitModelAttri(config);

        // 创建模型着色器
        model->CreateShader();

        LoadModelData(model_loaded, &model_loader, DEBUG_ASYNC_MODE);
        WaitForModelLoad(model_loaded);
        
        // 储存创建好的model指针
        MODEL_POINTERS.push_back(model);
        ValidateModelData(model);
        
        if(DEBUG_INFO_LV > 0) std::cout << "\n[INITER] 成功加载模型: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[INITER_ERROR] 模型加载失败: " << model_path << " - " << e.what() << std::endl;
        
        throw; // 向上传递异常以终止初始化
    }
}

// 加载模型数据的核心逻辑
void MIL::LoadModelData(std::atomic<bool>& model_loaded, ML* model_loader, bool async_mode) {
    if (async_mode) model_loader->LoadAsync([&]{ model_loaded.store(true); });
    else model_loader->LoadSync([&]{ model_loaded.store(true); });
}

// 监控加载状态的等待循环
void MIL::WaitForModelLoad(std::atomic<bool>& model_loaded) {
    auto start_time = csclock::now();
    auto& tracker = Loaders::ProgressTracker::Get();
    
    while (!model_loaded.load()) {
        int processed = 0;
        TaskQueue::ProcTasks(processed);
        
        CheckForTimeout(start_time);

        // 检查窗口关闭
        if (glfwWindowShouldClose(WINDOW::GetWindow())) {
            glfwTerminate();
            exit(EXIT_SUCCESS);
        }
        // 无任务时让出CPU
        if (processed == 0) {
            std::this_thread::sleep_for(millisec(10));
        }
    }
}

// 超时检测
void MIL::CheckForTimeout(const std::chrono::time_point<csclock>& start_time) {
    constexpr auto timeout = std::chrono::seconds(3);
    if (csclock::now() > start_time + timeout) {
        throw std::runtime_error("[INITER_ERROR] 模型加载超时");
    }
}

// 模型数据验证
void MIL::ValidateModelData(::CubeDemo::Model* model) {
    if (model->bounds.Rad < 0.01f) {
        std::cerr << "[INITER_ERROR] 模型包围球异常，可能未正确加载顶点数据" << std::endl;
        // 这里可以添加更多验证逻辑，比如检查顶点数量等
    }
}
} // namespace CubeDemo
