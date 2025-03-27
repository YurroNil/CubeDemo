// src/commandGenerator.cpp

#include "commandGenerator.h"

// g++命令、include库和静态链接库链接的 字符串拼接部分
string LinkingSpliSection() {

    vector<string> include = JsonParser::s_PathsInclude;
    vector<string> lib = JsonParser::s_PathsLib;
    vector<string> lib_suffix = JsonParser::s_argsLibSuffix;
    vector<string> buffer = {"", "", " "};

    for(string s : include) { buffer[0] += (" -I\"" + s + "\""); }
    for(string s : lib) { buffer[1] += (" -L\"" + s + "\""); }
    for(string s : lib_suffix) { buffer[2] += (s + " "); }

    return buffer[0] + buffer[1] + buffer[2];
}

// 构建PowerShell
string GeneratedFullCommands(vector<fs::path> &cpp_files) {

    // GCC命令
    string prefix{"powershell -Command '"}, buffer{""}, command{""};
    string gcc_command[4] = {
        "g++.exe -fexec-charset=utf-8 -g ",
        PathSpliSection(cpp_files),
        JsonParser::s_argsOutput,
        LinkingSpliSection()
    };
    // 字符串拼接
    for(string buffer : gcc_command) { command += buffer; }

    return command;
}