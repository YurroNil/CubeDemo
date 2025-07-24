// include/utils/string_conv.h
#pragma once

using wstring = std::wstring;

namespace CubeDemo {

class StringConvertor {
public:
    // UTF-8 转宽字符串 (wstring)
    static wstring U8_to_Wstring(const string& str);
    // 宽字符串 (wstring) 转 UTF-8
    static string Wstring_to_U8(const wstring& wstr);
    // UTF-8 转平台本地编码 (ANSI on Windows, UTF-8 elsewhere)
    static string U8_to_Native(const string& str);
    // 平台本地编码转 UTF-8
    static string Native_to_U8(const string& str);
};
}
