// include/utils/json_config.cpp
#include "pch.h"

namespace CubeDemo::Utils {

json JsonConfig::GetFileData(const string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + config_path);
    }
    json config;
    file >> config;
    return config;
}

// 十六进制字符串转ImWchar
ImWchar JsonConfig::HexStringToImWchar(const string& hex_str) {
    unsigned int value;
    std::stringstream ss;
    ss << std::hex << hex_str;
    ss >> value;
    return static_cast<ImWchar>(value);
}
// 加载字体配置
FontConfig JsonConfig::LoadFontConfig(const string& config_path) {
    
    json config = GetFileData(config_path);

    FontConfig font_config;
    
    // 解析模式开关
    font_config.custom_mode = config.value("CustomMode", true);

    // 解析自定义字符
    if (font_config.custom_mode) {
        for (const auto& line : config["CustomChars"]) {
            font_config.custom_chars.push_back(line.get<string>());
        }
    }
    // 解析Unicode范围
    else {
        const auto& ranges = config["ImWideCharRanges"];
        
        // 基础拉丁字符
        for (const auto& hex_str : ranges["basic_latin_chars"]) {
            font_config.unicode_ranges.push_back(HexStringToImWchar(hex_str));
        }
        
        // 自定义字符集
        for (const auto& hex_str : ranges["custom_char_sets"]) {
            font_config.unicode_ranges.push_back(HexStringToImWchar(hex_str));
        }
        
        // 结束符
        font_config.unicode_ranges.push_back(0);
    }

    return font_config;
}

// 解析模型配置中，属性列表中的数据
void JsonConfig::AnalyzeModelAttri(const auto& model, std::vector<ModelConfig>& model_config_array) {

    ModelConfig model_config;

    model_config.id = model.value("id", "");
    model_config.type = model.value("type", "");
    model_config.name = model.value("name", "");
    model_config.path = model.value("path", "");
    
    // 解析着色器路径
    if (model.find("shaders") != model.end()) {

        const auto& shaders = model["shaders"];
        model_config.vsh_path = shaders.value("vertex", "");
        model_config.fsh_path = shaders.value("fragment", "");
    }

    // 解析属性
    if (model.find("attributes") != model.end()) {

        const auto& attr = model["attributes"];

        // 位置
        auto pos = attr["position"];
        model_config.position = vec3(
            pos[0].template get<float>(),
            pos[1].template get<float>(),
            pos[2].template get<float>()
        );

        // 旋转（绕Y轴）
        model_config.rotation = attr.value("rotation", 0.0f);

        // 缩放
        auto scale_ = attr["scale"];
        model_config.scale = vec3(
            scale_[0].template get<float>(),
            scale_[1].template get<float>(),
            scale_[2].template get<float>()
        );
    }
    model_config_array.push_back(model_config);
}

// 加载模型配置
std::vector<ModelConfig> JsonConfig::LoadModelConfig(const string& config_path) {
    json config = GetFileData(config_path);
    std::vector<ModelConfig> model_configs;
    
    if (config.find("LoadModels") != config.end()) {
        for (const auto& model : config["LoadModels"]) {
            AnalyzeModelAttri(model, model_configs);
        }
    }
    return model_configs;
}
}   // namespace CubeDemo::Utils
