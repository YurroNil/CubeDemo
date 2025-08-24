// include/utils/graphic_drawing.inl
#pragma once

namespace CubeDemo::Utils {

inline void AddRectFilledRounded(ImDrawList* draw_list, const ImVec2& p_min, const ImVec2& p_max, ImU32 col, float rounding, ImDrawFlags flags = 0) {
    if (rounding > 0.0f) {
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_min.y + rounding), rounding, IM_PI, IM_PI * 1.5f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_min.y + rounding), rounding, IM_PI * 1.5f, IM_PI * 2.0f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_max.y - rounding), rounding, 0.0f, IM_PI * 0.5f);
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_max.y - rounding), rounding, IM_PI * 0.5f, IM_PI);
        draw_list->PathFillConvex(col);
    } else {
        draw_list->AddRectFilled(p_min, p_max, col, 0.0f);
    }
}

// 兼容旧版ImGui的圆角边框绘制函数
inline void AddRectRounded(ImDrawList* draw_list, const ImVec2& p_min, const ImVec2& p_max, ImU32 col, float rounding, float thickness, ImDrawFlags flags = 0) {
    if (rounding > 0.0f) {
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_min.y + rounding), rounding, IM_PI, IM_PI * 1.5f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_min.y + rounding), rounding, IM_PI * 1.5f, IM_PI * 2.0f);
        draw_list->PathArcTo(ImVec2(p_max.x - rounding, p_max.y - rounding), rounding, 0.0f, IM_PI * 0.5f);
        draw_list->PathArcTo(ImVec2(p_min.x + rounding, p_max.y - rounding), rounding, IM_PI * 0.5f, IM_PI);
        draw_list->PathStroke(col, true, thickness);
        
    } else draw_list->AddRect(p_min, p_max, col, 0.0f, 0, thickness);
}
}