// include/loaders/fonts.h
#pragma once
#include "threads/loaders.h"
#include "utils/jsonConfig.h"

using FL = CubeDemo::Loaders::Fonts;
using IFGRB = ImFontGlyphRangesBuilder;

namespace CubeDemo {

class Loaders::Fonts {
public:
    static void BuildFromUnicodeRanges(IFGRB& builder, const std::vector<ImWchar>& ranges);
    static void BuildFromCustomChars(IFGRB& builder, const std::vector<string>& char_lines);
    static void LoadFonts();
};
}