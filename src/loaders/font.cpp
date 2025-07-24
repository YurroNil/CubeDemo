// include/loaders/fonts.cpp
#include "pch.h"
#include "loaders/font.h"
#include "loaders/progress_tracker.h"
#include "kits/file_system.h"
#include "utils/defines.h"
#include "utils/font_defines.h"

// 外部函数声明
extern int utf8_conv_unicode(unsigned int* out_char, const char* in_text, const char* in_text_end);
namespace CubeDemo {

// 辅助函数：加载带中文字体合并的字体
ImFont* FL::AddFontWithMergedCN(
    const char* en_font_path, 
    float size, 
    const char* cn_font_path,
    const ImVector<ImWchar>& cn_ranges,
    const char* fa_font_path,
    const ImWchar* fa_ranges,
    float cn_offset_y,
    float fa_offset_y
) {
    ImGuiIO& io = ImGui::GetIO();
    
    // 加载基础字体（英文字体）
    ImFont* font = io.Fonts->AddFontFromFileTTF(
        en_font_path,
        size,
        nullptr,
        io.Fonts->GetGlyphRangesDefault()
    );
    
    if (!font) {
        std::cerr << "无法加载英文字体: " << en_font_path << std::endl;
        return nullptr;
    }
    
    // 合并中文字体
    if (fs::exists(cn_font_path)) {
        ImFontConfig cn_config;
        cn_config.MergeMode = true;
        cn_config.GlyphOffset.y = cn_offset_y;
        
        io.Fonts->AddFontFromFileTTF(
            cn_font_path,
            size, // 中文字体大小与英文字体相同
            &cn_config,
            cn_ranges.Data
        );
    } else std::cerr << "中文字体文件不存在: " << cn_font_path << std::endl;
    
    // 合并图标字体
    if (fs::exists(fa_font_path)) {
        ImFontConfig fa_config;
        fa_config.MergeMode = true;
        fa_config.GlyphOffset.y = fa_offset_y;
        fa_config.GlyphMinAdvanceX = 20.0f;
        fa_config.PixelSnapH = true;
        
        io.Fonts->AddFontFromFileTTF(
            fa_font_path,
            size * 0.9f, // 图标字体稍小
            &fa_config,
            fa_ranges
        );
    } else std::cerr << "图标字体文件不存在: " << fa_font_path << std::endl;
    
    return font;
}

// 加载自定义字体
void FL::LoadFonts() {
    // 添加字体加载任务
    ProgressTracker::Get().AddResource(
        ProgressTracker::FONT, 
        "fonts"
    );

    ImGuiIO& io = ImGui::GetIO();
    ImFontGlyphRangesBuilder builder;

    // 构建中文字符范围
    builder.Clear();
    auto font_config = Utils::JsonConfig::LoadFontConfig(FONT_PATH + string("custom_chars.json"));
    
    // 模式选择：优先使用Unicode范围
    if (font_config.custom_mode) CustomChars(builder, font_config.custom_chars);
    else UnicodeRanges(builder, font_config.unicode_ranges);

    ImVector<ImWchar> cn_ranges;
    builder.BuildRanges(&cn_ranges);

    // 定义字体路径
    const char* en_font_path = "C:/Windows/Fonts/consola.ttf";
    const char* cn_font_path = "C:/Windows/Fonts/Deng.ttf";
    string fa_path_str = FONT_PATH + string("Font Awesome 6 Free-Solid-900.otf");
    const char* fa_font_path = fa_path_str.c_str();
    
    // 图标范围定义
    static const ImWchar fa_ranges[] = {
        ICON_MIN_FA, ICON_MAX_FA, // 完整图标范围
        0
    };

    // 加载不同尺寸的字体
    // 大标题字体 (100px)
    largeTitleFont = AddFontWithMergedCN(
        en_font_path, 
        100.0f, 
        cn_font_path,
        cn_ranges,
        fa_font_path,
        fa_ranges,
        4.0f,  // 中文垂直偏移
        -4.0f  // 图标垂直偏移
    );
    
    // 副标题字体 (50px)
    subtitleFont = AddFontWithMergedCN(
        en_font_path, 
        50.0f, 
        cn_font_path,
        cn_ranges,
        fa_font_path,
        fa_ranges,
        1.5f,  // 中文垂直偏移
        -1.5f  // 图标垂直偏移
    );
    
    // 标题字体 (40px)
    headerFont = AddFontWithMergedCN(
        en_font_path, 
        40.0f, 
        cn_font_path,
        cn_ranges,
        fa_font_path,
        fa_ranges,
        1.0f,  // 中文垂直偏移
        -1.0f  // 图标垂直偏移
    );
    
    // 默认字体 (30px)
    defaultFont = AddFontWithMergedCN(
        en_font_path, 
        30.0f, 
        cn_font_path,
        cn_ranges,
        fa_font_path,
        fa_ranges,
        0.5f,  // 中文垂直偏移
        -0.5f  // 图标垂直偏移
    );
    
    // 设置默认字体. 优先级：默认字体 > 标题字体 > 副标题字体 > 大标题字体
    io.FontDefault = defaultFont ? defaultFont : headerFont ? headerFont : subtitleFont ? subtitleFont : largeTitleFont;
    

    // 构建字体集
    // io.Fonts->Build();

    // 更新进度：字体加载完成
    ProgressTracker::Get().FinishResource(
        ProgressTracker::FONT, 
        "fonts"
    );
}

// 字体获取接口实现
ImFont* FL::GetLargeTitleFont() { 
    return largeTitleFont ? largeTitleFont : GetDefaultFont(); 
}

ImFont* FL::GetSubtitleFont() { 
    return subtitleFont ? subtitleFont : GetDefaultFont(); 
}

ImFont* FL::GetHeaderFont() { 
    return headerFont ? headerFont : GetDefaultFont(); 
}

ImFont* FL::GetDefaultFont() { 
    return defaultFont ? defaultFont : ImGui::GetFont(); 
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
