// include/utils/jsonConfig.h
#pragma once
#include "pch.h"

using json = nlohmann::json;

namespace CubeDemo::Utils {
using StringArray = std::vector<string>;


struct FontConfig {
    bool custom_mode;
    StringArray custom_chars;
    std::vector<ImWchar> unicode_ranges;
};

class JsonConfig {
public:
    static json GetFileData(const string& config_path);

    static StringArray LoadModelList(const string& config_path);
    // 新增字体配置加载
    static FontConfig LoadFontConfig(const string& config_path);

private:
    static ImWchar HexStringToImWchar(const string& hex_str);
};

}
