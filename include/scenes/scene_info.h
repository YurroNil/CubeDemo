#pragma once

namespace CubeDemo::Scenes {

struct ScenePrefab {
    string type;
    string path;
};

struct SceneInfo {
    string id;
    string name;
    string description;
    string author;
    string icon;
    std::vector<ScenePrefab> prefabs;
    
    static SceneInfo FromJson(const json& json);
};

} // namespace CubeDemo::Scenes