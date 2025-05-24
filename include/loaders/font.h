// include/loaders/font.h
#pragma once
#include "threads/loaders.h"
#include "utils/jsonConfig.h"

using FL = CubeDemo::Loaders::Font;
using IFGRB = ImFontGlyphRangesBuilder;

namespace CubeDemo {

class Loaders::Font {
public:
    static void UnicodeRanges(IFGRB& builder, const std::vector<ImWchar>& ranges);
    static void CustomChars(IFGRB& builder, const std::vector<string>& char_lines);
    static void LoadFonts();
};
}