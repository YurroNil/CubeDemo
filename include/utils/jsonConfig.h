// include/utils/jsonConfig.h
#pragma once
#include "kits/strings.h"
#include <vector>
#include "kits/imgui.h"

namespace CubeDemo::Utils {
using StringArray = std::vector<string>;

struct FontConfig {
    bool custom_mode;
    StringArray custom_chars;
    std::vector<ImWchar> unicode_ranges;
};

class JsonConfig {
public:
    static StringArray LoadModelList(const string& config_path);
    // 新增字体配置加载
    static FontConfig LoadFontConfig(const string& config_path);
    
private:
    static ImWchar HexStringToImWchar(const string& hex_str);
};

}
