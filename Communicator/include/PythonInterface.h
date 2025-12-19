#ifndef PYTHON_INTERFACE_H
#define PYTHON_INTERFACE_H

#include <vector>
#include "StructuredData.h"

// Python接口类 - 预留与Python进行数据对接
// 可以通过C API或pybind11等方式实现
class PythonInterface {
public:
    PythonInterface();
    ~PythonInterface();

    // 初始化Python环境
    bool Initialize();
    
    // 关闭Python环境
    void Shutdown();
    
    // 发送数据到Python层（异常检测层）
    bool SendToPythonDetection(const StructuredData& data);
    
    // 批量发送数据到Python层
    bool SendToPythonDetectionBatch(const std::vector<StructuredData>& data_list);
    
    // 从Python层接收数据（可选，用于双向通信）
    bool ReceiveFromPython(std::vector<StructuredData>& data_list);
    
    // 检查Python环境是否可用
    bool IsAvailable() const;

private:
    class PythonInterfaceImpl;
    std::unique_ptr<PythonInterfaceImpl> impl_;
    bool is_initialized_;
};

// C接口 - 用于Python C API调用
extern "C" {
    // 创建Python接口实例
    void* CreatePythonInterface();
    
    // 销毁Python接口实例
    void DestroyPythonInterface(void* handle);
    
    // 发送数据到Python
    int SendDataToPython(void* handle, const StructuredData* data);
}

#endif // PYTHON_INTERFACE_H

