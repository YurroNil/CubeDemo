#include "pch.h"
#include "managers/scene/mng.h"
#include "scenes/base.h"
#include "utils/json_config.h"
#include "scenes/dynamic_scene.h"

namespace fs = std::filesystem;

namespace CubeDemo::Managers {

void SceneMng::Init(const string& sceneDir) {
    // 初始化场景管理器
    m_sceneDir = sceneDir;
    // 加载场景配置
    LoadSceneConfigs();
    
    // 设置默认场景
    // 如果默认场景存在，则切换到默认场景
    if (m_scenes.find(m_defaultSceneID) != m_scenes.end()) {
        SwitchTo(m_defaultSceneID);
    // 如果默认场景不存在，但场景列表不为空，则切换到第一个场景
    } else if (!m_scenes.empty()) {
        SwitchTo(m_scenes.begin()->first);
    }
}

// 加载场景配置
void SceneMng::LoadSceneConfigs() {
    // 遍历场景目录
    for (const auto& entry : fs::directory_iterator(m_sceneDir)) {
        // 如果是目录
        if (entry.is_directory()) {
            // 加载场景
            LoadScene(entry.path());
        }
    }
}

void SceneMng::LoadScene(const fs::path& sceneDir) {
    // 获取场景ID
    string sceneID = sceneDir.filename().string();
    
    // 避免重复加载
    if (m_scenes.find(sceneID) != m_scenes.end()) return;
    
    // 获取场景配置文件路径
    fs::path configPath = sceneDir / "scene_info.json";
    // 检查配置文件是否存在
    if (!fs::exists(configPath)) {
        std::cerr << "场景配置文件不存在: " << configPath << std::endl;
        return;
    }
    
    try {
        // 读取配置文件
        json config = Utils::JsonConfig::GetFileData(configPath.string());
        // 解析配置文件
        Scenes::SceneInfo info = Scenes::SceneInfo::FromJson(config);
        // 设置资源路径
        info.resourcePath = sceneDir.string();
        
        // 创建动态场景
        m_scenes[sceneID] = new Scenes::DynamicScene(info);
    } catch (const std::exception& e) {
        std::cerr << "加载场景失败: " << sceneDir << " - " << e.what() << std::endl;
    }
}

void SceneMng::SwitchTo(const string& sceneID) {
    // 在m_scenes中查找sceneID
    auto it = m_scenes.find(sceneID);
    // 如果找不到sceneID，则输出错误信息并返回
    if (it == m_scenes.end()) {
        std::cerr << "场景不存在: " << sceneID << std::endl;
        return;
    }
    
    // 将当前场景设置为找到的场景
    m_currentScene = it->second;
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
        delete scene; // 删除场景
    }
    m_scenes.clear(); // 清空场景列表
}

} // namespace CubeDemo::Managers