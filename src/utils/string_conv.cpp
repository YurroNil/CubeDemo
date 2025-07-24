// src/utils/string_conv.cpp
#include "pch.h"
#include "utils/string_conv.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <locale>
#include <codecvt>
#endif

namespace CubeDemo {

#ifdef _WIN32
// Windows实现
wstring StringConvertor::U8_to_Wstring(const string& str) {
    if (str.empty()) return L"";
    
    int size_needed = MultiByteToWideChar(
        CP_UTF8, 0, 
        str.c_str(), (int)str.size(), 
        NULL, 0
    );
    
    wstring wstr(size_needed, 0);
    MultiByteToWideChar(
        CP_UTF8, 0, 
        str.c_str(), (int)str.size(), 
        &wstr[0], size_needed
    );
    
    return wstr;
}

string StringConvertor::Wstring_to_U8(const wstring& wstr) {
    if (wstr.empty()) return "";
    
    int size_needed = WideCharToMultiByte(
        CP_UTF8, 0,
        wstr.c_str(), (int)wstr.size(),
        NULL, 0, NULL, NULL
    );
    
    string str(size_needed, 0);
    WideCharToMultiByte(
        CP_UTF8, 0,
        wstr.c_str(), (int)wstr.size(),
        &str[0], size_needed,
        NULL, NULL
    );
    
    return str;
}

string StringConvertor::U8_to_Native(const string& str) {
    // Windows本地编码是ANSI
    wstring wstr = U8_to_Wstring(str);
    
    int size_needed = WideCharToMultiByte(
        CP_ACP, 0,
        wstr.c_str(), (int)wstr.size(),
        NULL, 0, NULL, NULL
    );
    
    string native(size_needed, 0);
    WideCharToMultiByte(
        CP_ACP, 0,
        wstr.c_str(), (int)wstr.size(),
        &native[0], size_needed,
        NULL, NULL
    );
    
    return native;
}

string StringConvertor::Native_to_U8(const string& str) {
    // Windows本地编码是ANSI
    int size_needed = MultiByteToWideChar(
        CP_ACP, 0,
        str.c_str(), (int)str.size(),
        NULL, 0
    );
    
    wstring wstr(size_needed, 0);
    MultiByteToWideChar(
        CP_ACP, 0,
        str.c_str(), (int)str.size(),
        &wstr[0], size_needed
    );
    
    return Wstring_to_U8(wstr);
}

#else
// macOS/Linux 实现 (使用标准库)

wstring StringConvertor::U8_to_Wstring(const string& str) {
    // 禁用弃用警告
    #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #endif
    
    wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    auto result = converter.from_bytes(str);
    
    #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC diagnostic pop
    #endif
    
    return result;
}

string StringConvertor::Wstring_to_U8(const wstring& wstr) {
    #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #endif
    
    wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    auto result = converter.to_bytes(wstr);
    
    #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC diagnostic pop
    #endif
    
    return result;
}

// 在非Windows系统上，本地编码通常是UTF-8
string StringConvertor::U8_to_Native(const string& str) {
    return str; // 直接返回，因为本地编码就是UTF-8
}

string StringConvertor::Native_to_U8(const string& str) {
    return str; // 直接返回，因为本地编码就是UTF-8
}

#endif

} // namespace CubeDemo
