// include/utils/json_config.h
#pragma once

using json = nlohmann::json;

namespace CubeDemo::Utils {
using StringArray = std::vector<string>;

struct FontConfig {
    bool custom_mode;
    StringArray custom_chars;
    std::vector<ImWchar> unicode_ranges;
};

struct ModelConfig {
    string id, name, type, path;
    vec3 position = vec3(0.0f);
    float rotation = 0.0f; // 绕Y轴旋转角度（度）
    vec3 scale = vec3(1.0f);
};

class JsonConfig {

    static ImWchar HexStringToImWchar(const string& hex_str);

public:
    static json GetFileData(const string& config_path);
    
    // 字体配置加载
    static FontConfig LoadFontConfig(const string& config_path);

    // 模型配置加载
    static std::vector<ModelConfig> LoadModelConfig(const string& config_path);

    // 解析模型配置中，属性列表中的数据
    static void AnalyzeModelAttri(const auto& model, std::vector<ModelConfig>& model_config_array);
};
}

