// src/utils/UTF8_to_unicode.cpp

int utf8_conv_unicode(unsigned int* out_char, const char* in_text, const char* in_text_end) {
    if (!in_text || !out_char) return 0;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(in_text);
    if (in_text_end && p >= reinterpret_cast<const unsigned char*>(in_text_end)) return 0;

    // ASCII 快速路径
    if (*p < 0x80) {
        *out_char = *p;
        return 1;
    }

    // 多字节解码
    unsigned int codepoint = 0;
    int bytes = 0;
    if      ((*p & 0xE0) == 0xC0) { bytes = 2; codepoint = *p & 0x1F; }
    else if ((*p & 0xF0) == 0xE0) { bytes = 3; codepoint = *p & 0x0F; }
    else if ((*p & 0xF8) == 0xF0) { bytes = 4; codepoint = *p & 0x07; }
    else return 0;

    // 检查输入长度
    if (in_text_end && (p + bytes > reinterpret_cast<const unsigned char*>(in_text_end)))
        return 0;

    // 解码后续字节
    for (int i = 1; i < bytes; ++i) {
        if ((p[i] & 0xC0) != 0x80)
            return 0;
        codepoint = (codepoint << 6) | (p[i] & 0x3F);
    }

    // 验证码点范围（Unicode 规范）
    if (codepoint > 0x10FFFF || 
        (bytes == 2 && codepoint < 0x80) ||
        (bytes == 3 && codepoint < 0x800) ||
        (bytes == 4 && codepoint < 0x10000)) {
        return 0;
    }

    *out_char = codepoint;
    return bytes;
}
