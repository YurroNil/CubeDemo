// include/utils/json_config.h
#pragma once
#include "utils/defines.h"

namespace CubeDemo::Utils {
using StringArray = std::vector<string>;

struct FontConfig {
    bool custom_mode;
    StringArray custom_chars;
    std::vector<ImWchar> unicode_ranges;
};

struct ModelConfig {

    string id = "unknown", name = "未命名模型", type = "model.unknown", path = "null";

    vec3 position = vec3(0.0f), scale = vec3(1.0f), rotation = vec3(0.0f);

    string
        vsh_path = VSH_PATH + string("model.glsl"),
        fsh_path = FSH_PATH + string("model_none.glsl"),
        icon_path = ICON_PATH + string("unknown.png"), description = "无描述.";
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
    static void AnalyzeModelAttri(
        const auto& model,
        std::vector<ModelConfig>& model_config_array
    );
};
}   // namespace CubeDemo::Utils
