// include/utils/jsonConfig.cpp
#include "utils/jsonConfig.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace CubeDemo::Utils {

StringArray JsonConfig::LoadModelList(const string& config_path) {

    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + config_path);
    }

    nlohmann::json config;
    file >> config;

    StringArray models;
    for (const auto& model_name : config["LoadModels"]) {
        models.push_back(model_name.get<string>());
    }

    return models;
}

// 十六进制字符串转ImWchar
ImWchar JsonConfig::HexStringToImWchar(const string& hex_str) {
    unsigned int value;
    std::stringstream ss;
    ss << std::hex << hex_str;
    ss >> value;
    return static_cast<ImWchar>(value);
}

FontConfig JsonConfig::LoadFontConfig(const string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开字体配置文件: " + config_path);
    }

    nlohmann::json config;
    file >> config;

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




}   // namespace CubeDemo::Utils
