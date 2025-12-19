#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <string>
#include <map>

class ConfigManager {
public:
    static ConfigManager& GetInstance();
    
    // 从文件加载配置
    bool LoadFromFile(const std::string& config_path);
    
    // 获取配置值
    std::string GetString(const std::string& key, const std::string& default_value = "");
    int GetInt(const std::string& key, int default_value = 0);
    double GetDouble(const std::string& key, double default_value = 0.0);
    bool GetBool(const std::string& key, bool default_value = false);
    
    // 设置配置值（运行时修改）
    void SetString(const std::string& key, const std::string& value);
    void SetInt(const std::string& key, int value);
    void SetDouble(const std::string& key, double value);
    void SetBool(const std::string& key, bool value);

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    std::map<std::string, std::string> config_map_;
    std::mutex mutex_;
};

#endif // CONFIG_MANAGER_H

