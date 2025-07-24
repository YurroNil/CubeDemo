// src/utils/time.cpp
#include "pch.h"
#include <ctime>
#include "utils/time.h"

namespace CubeDemo::Utils::Time {

// 获取当前时间对应的问候语
string get_time_greeting() {
    time_t now = time(0);
    tm ltm;
    localtime_s(&ltm, &now);
    int hour = ltm.tm_hour;

    if (hour >= 5 && hour < 12) {
        return "早上好";
    } else if (hour >= 12 && hour < 14) {
        return "中午好";
    } else if (hour >= 14 && hour < 18) {
        return "下午好";
    } else if (hour >= 18 && hour < 22) {
        return "晚上好";
    } else {
        return "夜深了";
    }
}
} // namespace CubeDemo::Utils::Time
