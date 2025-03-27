// include/include.h

#pragma once

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>
#include "nlohmann/json.hpp"

using namespace std;
namespace fs = std::filesystem;
using string = std::string;
using n_json = nlohmann::json;

class JsonParser {
public:
    JsonParser() = default;
    ~JsonParser() = default;

    static vector<string> s_PathsInclude;
    static vector<string> s_PathsLib;
    static vector<string> s_PathsSourceCode;
    static vector<string> s_argsLibSuffix;
    static string s_argsOutput;
    static int s_argsCount;

    static void LoadFromJson(const string& filePath);

    
private:
    static void Parsing(n_json& data);

};