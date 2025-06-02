// include/loaders/fonts.cpp

#include "loaders/font.h"
#include "kits/file_system.h"


// 外部函数声明
extern int utf8_to_unicode_conv(unsigned int* out_char, const char* in_text, const char* in_text_end);

namespace CubeDemo {

// 加载自定义字体
void FL::LoadFonts() {
    constexpr const char* FONT_CONFIG_PATH = "../resources/fonts/custom_chars.json";
    constexpr const char* FONT_FILE_PATH = "../resources/fonts/simhei.ttf";

    // 检查字体文件存在性
    if (!fs::exists(FONT_FILE_PATH)) {
        std::cerr << "字体文件不存在: " << FONT_FILE_PATH << std::endl;
        return;
    }

    try {
        // 加载字体配置
        auto font_config = Utils::JsonConfig::LoadFontConfig(FONT_CONFIG_PATH);
        
        ImGuiIO& io = ImGui::GetIO();
        ImFontGlyphRangesBuilder builder;

        // 模式选择
        if (font_config.custom_mode) {
            CustomChars(builder, font_config.custom_chars);
        } else {
            UnicodeRanges(builder, font_config.unicode_ranges);
        }

        // 生成紧凑字符范围
        ImVector<ImWchar> ranges;
        builder.BuildRanges(&ranges);

        // 加载字体
        ImFont* font = io.Fonts->AddFontFromFileTTF(
            FONT_FILE_PATH,
            30.0f,
            nullptr,
            ranges.Data
        );
        
        io.Fonts->Build();
        std::cout << "成功加载字体: " << FONT_FILE_PATH << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "字体加载失败: " << e.what() << std::endl;
    }
}

void FL::CustomChars(IFGRB& builder, const std::vector<string>& char_lines) {
    // 添加基础拉丁字符
    builder.AddRanges(ImGui::GetIO().Fonts->GetGlyphRangesDefault());

    // 添加自定义字符
    for (const auto& line : char_lines) {
        const char* p = line.c_str();
        while (*p) {
            unsigned int c = 0;
            int bytes = utf8_to_unicode_conv(&c, p, nullptr);
            if (bytes == 0) break;
            builder.AddChar(static_cast<ImWchar>(c));
            p += bytes;
        }
    }
}

// 从Unicode范围构建字符集
void FL::UnicodeRanges(IFGRB& builder, const std::vector<ImWchar>& ranges) {
    // 直接添加预定义的Unicode范围
    for (size_t i = 0; i < ranges.size(); ) {
        if (ranges[i] == 0) break;
        builder.AddRanges(&ranges[i]);
        i += 2; // 跳过范围对
    }
}
}
