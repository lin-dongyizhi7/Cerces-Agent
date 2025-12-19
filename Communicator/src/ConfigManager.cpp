#include "ConfigManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

ConfigManager& ConfigManager::GetInstance() {
    static ConfigManager instance;
    return instance;
}

bool ConfigManager::LoadFromFile(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // 解析 key=value 格式
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // 去除首尾空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            config_map_[key] = value;
        }
    }
    
    file.close();
    std::cout << "Loaded " << config_map_.size() << " config entries from " << config_path << std::endl;
    return true;
}

std::string ConfigManager::GetString(const std::string& key, const std::string& default_value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = config_map_.find(key);
    return (it != config_map_.end()) ? it->second : default_value;
}

int ConfigManager::GetInt(const std::string& key, int default_value) {
    std::string value = GetString(key);
    if (value.empty()) {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return default_value;
    }
}

double ConfigManager::GetDouble(const std::string& key, double default_value) {
    std::string value = GetString(key);
    if (value.empty()) {
        return default_value;
    }
    try {
        return std::stod(value);
    } catch (...) {
        return default_value;
    }
}

bool ConfigManager::GetBool(const std::string& key, bool default_value) {
    std::string value = GetString(key);
    if (value.empty()) {
        return default_value;
    }
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return (value == "true" || value == "1" || value == "yes");
}

void ConfigManager::SetString(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_[key] = value;
}

void ConfigManager::SetInt(const std::string& key, int value) {
    SetString(key, std::to_string(value));
}

void ConfigManager::SetDouble(const std::string& key, double value) {
    SetString(key, std::to_string(value));
}

void ConfigManager::SetBool(const std::string& key, bool value) {
    SetString(key, value ? "true" : "false");
}

