#include "pch.h"
#include "managers/scene.h"
#include "scenes/base.h"
#include "utils/json_config.h"
#include "scenes/dynamic_scene.h"
#include "loaders/model_initer.h"
#include "ui/screens/loading.h"

namespace fs = std::filesystem;

namespace CubeDemo {
    extern bool RAY_TRACING_ENABLED, RT_DEBUG;
}

namespace CubeDemo::Managers {

void SceneMng::Init(const string& sceneDir) {
    m_sceneDir = sceneDir;
    LoadSceneConfigs();
}

void SceneMng::LoadSceneConfigs() {
    for (const auto& entry : fs::directory_iterator(m_sceneDir)) {
        if (!entry.is_directory()) continue;
        ParsingData(entry.path());
    }
}

// 解析json文件，然后将数据存进SceneInfo中
void SceneMng::ParsingData(const fs::path& sceneDir) {
    
    string sceneID = sceneDir.filename().string();
    if (m_scenes.find(sceneID) != m_scenes.end()) return;
    fs::path configPath = sceneDir / "scene_info.json";

    if (!fs::exists(configPath)) {
        std::cerr << "错误: 场景配置文件不存在: " << configPath << std::endl;
        return;
    }
    
    try {
        // 读取文件内容
        std::ifstream file(configPath);
        string content(
            (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()
        );
        
        // 解析JSON
        json config = json::parse(content);
        
        // 创建场景信息
        Scenes::SceneInfo info = Scenes::SceneInfo::FromJson(config);
        info.resourcePath = sceneDir.string();
        
        // 添加预览图路径解析
        if (config.contains("preview_image")) {
            info.previewImage = config["preview_image"].get<string>();
        }
        
        // 解析完成后，创建DynamicScene实例来管理多个场景.
        m_scenes[sceneID] = new Scenes::DynamicScene(info);
        
    } catch (const std::exception& e) {
        std::cerr << "加载场景失败: " << sceneDir << " - " << e.what() << std::endl;
    }
}

// 仅卸载场景资源
void SceneMng::Cleanup() {
    if (m_currentScene) m_currentScene->Cleanup();
    m_currentScene = nullptr;
}

// 切换场景
void SceneMng::SwitchTo(const string& sceneID) {
    auto it = m_scenes.find(sceneID);
    if (it == m_scenes.end()) {
        std::cerr << "场景不存在: " << sceneID << std::endl;
        return;
    }
    // 先卸载当前场景, 再加载新场景
    Cleanup(); m_currentScene = it->second;
    // 初始化场景资源
    try {
        // 初始化模型资源
        MIL::InitModels();
    }
    // 失败处理
    catch (const std::exception& e) {
        std::cerr << "场景初始化失败: " << sceneID << " - " << e.what() << std::endl;
        m_currentScene = nullptr;
    }
}

SceneMng* SceneMng::CreateInst() {
    if(m_InstCount > 0) {
        std::cerr << "[SceneMng] 场景创建失败，因为当前场景管理器数量为: " << m_InstCount << std::endl;
        return nullptr;
    }
    m_InstCount++;
    return new SceneMng();
}

void SceneMng::RemoveInst(SceneMng** ptr) {
    if (ptr && *ptr) {
        delete *ptr;
        *ptr = nullptr;
        if (m_InstCount > 0) {
            m_InstCount--;
        }
    }
}

SceneMng::~SceneMng() {
    // 清理所有场景
    for (auto& [id, scene] : m_scenes) {
        scene->Cleanup();
        delete scene; // 删除场景
    }
    m_scenes.clear(); // 清空场景列表
}
} // namespace CubeDemo::Managers
