// src/utils/stringConvert.cpp

#include "utils/stringConvertor.h"

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