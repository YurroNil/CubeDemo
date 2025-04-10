// src/utils/stringConvert.cpp
#include <codecvt>  // 用于wstring_convert
#include <locale>   // 用于wstring_convert
#include "utils/stringConvertor.h"
namespace CubeDemo {

// utf8转换成wstring
wstring StringConvertor::U8_to_Wstring(const string& str) {

    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}
 
 // wstring转换成wtf8
string StringConvertor::WstringTo_U8(const wstring& wstr) {

    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

}