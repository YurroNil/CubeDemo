// include/loaders/fonts.cpp
#include "pch.h"
#include "loaders/font.h"
#include "kits/file_system.h"
#include "utils/defines.h"

// 外部函数声明
extern int utf8_conv_unicode(unsigned int* out_char, const char* in_text, const char* in_text_end);
namespace CubeDemo {

// 加载自定义字体
void FL::LoadFonts() {
    ImGuiIO& io = ImGui::GetIO();
    ImFontGlyphRangesBuilder builder;

    // 1. 加载英文字体 (Consola)
    const char* en_font_path = "C:/Windows/Fonts/consola.ttf";
    if (!fs::exists(en_font_path)) {
        std::cerr << "英文字体文件不存在: " << en_font_path << std::endl;
        return;
    }

    // 英文字符范围：基础拉丁字符
    builder.AddRanges(io.Fonts->GetGlyphRangesDefault()); 
    ImVector<ImWchar> en_ranges;
    builder.BuildRanges(&en_ranges);

    ImFont* en_font = io.Fonts->AddFontFromFileTTF(
        en_font_path,
        30.0f,
        nullptr,
        en_ranges.Data // 仅加载英文所需字符
    );

    // 2. 加载中文字体
    const char* cn_font_path = "C:/Windows/Fonts/Deng.ttf";
    if (!fs::exists(cn_font_path)) {
        std::cerr << "中文字体文件不存在: " << cn_font_path << std::endl;
        return;
    }

    // 重置Builder并添加中文字符范围
    builder.Clear();
    auto font_config = Utils::JsonConfig::LoadFontConfig(FONT_PATH + string("custom_chars.json"));
    
    // 模式选择：优先使用Unicode范围
    if (font_config.custom_mode) {
        CustomChars(builder, font_config.custom_chars);
    } else {
        UnicodeRanges(builder, font_config.unicode_ranges);
    }

    ImVector<ImWchar> cn_ranges;
    builder.BuildRanges(&cn_ranges);

    // 关键：启用MergeMode合并到基础字体
    ImFontConfig cn_config;
    cn_config.MergeMode = true; // 设置为合并模式
    cn_config.GlyphOffset.y = 2.0f; // 微调中文垂直偏移

    io.Fonts->AddFontFromFileTTF(
        cn_font_path,
        30.0f,
        &cn_config,
        cn_ranges.Data // 仅加载中文所需字符范围
    );

    // 3. 构建字体集
    io.Fonts->Build();
    std::cout << "字体合并完成: 英文(Consola) + 中文(等线)" << std::endl;
}

void FL::CustomChars(IFGRB& builder, const std::vector<string>& char_lines) {
    // 添加基础拉丁字符
    builder.AddRanges(ImGui::GetIO().Fonts->GetGlyphRangesDefault());

    // 添加自定义字符
    for (const auto& line : char_lines) {
        const char* p = line.c_str();
        while (*p) {
            unsigned int c = 0;
            int bytes = utf8_conv_unicode(&c, p, nullptr);
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
