// include/kits/times.h
#pragma once

#include <chrono>
using millisec = std::chrono::milliseconds;
using seconds = std::chrono::seconds;
using steady_clock = std::chrono::steady_clock;
using timepoint_steady = std::chrono::time_point<std::chrono::steady_clock>;
