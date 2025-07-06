// include/loaders/font.h
#pragma once
#include "loaders/fwd.h"
using FL = CubeDemo::Loaders::Font;
using IFGRB = ImFontGlyphRangesBuilder;

namespace CubeDemo {

class Loaders::Font {
public:
    static void UnicodeRanges(IFGRB& builder, const std::vector<ImWchar>& ranges);
    static void CustomChars(IFGRB& builder, const std::vector<string>& char_lines);
    static void LoadFonts();

    // 字体获取接口
    static ImFont* GetLargeTitleFont();
    static ImFont* GetSubtitleFont();
    static ImFont* GetHeaderFont();
    static ImFont* GetDefaultFont();

private:
    // 字体指针
    inline static ImFont* largeTitleFont = nullptr, *subtitleFont = nullptr, *headerFont = nullptr, *defaultFont = nullptr;
    
    // 加载特定字体
    static ImFont* AddFontWithMergedCN(
        const char* en_font_path, 
        float size, 
        const char* cn_font_path,
        const ImVector<ImWchar>& cn_ranges,
        const char* fa_font_path,
        const ImWchar* fa_ranges,
        float cn_offset_y = 2.0f,
        float fa_offset_y = -2.0f
    );

};
}   // namespace
