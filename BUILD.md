# 构建说明

## 依赖要求

### 系统要求
- Linux（CentOS 7+、Ubuntu 18.04+）或 macOS
- CMake 3.10+
- C++17编译器（GCC 7+或Clang 5+）

### 依赖库
- **Protobuf**：用于协议序列化/反序列化
  - Ubuntu/Debian: `sudo apt-get install libprotobuf-dev protobuf-compiler`
  - CentOS/RHEL: `sudo yum install protobuf-devel protobuf-compiler`
  - macOS: `brew install protobuf`

- **ZeroMQ**：用于消息队列
  - Ubuntu/Debian: `sudo apt-get install libzmq3-dev`
  - CentOS/RHEL: `sudo yum install zeromq-devel`
  - macOS: `brew install zeromq`

## 编译步骤

### 1. 克隆或下载代码

```bash
cd /path/to/Cerces-Agent
```

### 2. 创建构建目录

```bash
mkdir build
cd build
```

### 3. 运行CMake

```bash
cmake ..
```

如果需要指定安装路径：
```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
```

### 4. 编译

```bash
make
```

### 5. 安装（可选）

```bash
sudo make install
```

## 编译输出

编译完成后，在`build`目录下会生成以下可执行文件：
- `communicator`：通信转义层服务
- `agent`：采集端客户端

## 运行测试

### 1. 启动Communicator

在一个终端中：
```bash
cd build
./communicator ../config/communicator.conf
```

### 2. 启动Agent

在另一个终端中：
```bash
cd build
./agent localhost 8888 node_0 rank_0 1000
```

### 3. 验证

如果一切正常，Communicator应该会显示接收到的消息统计信息。

## 常见问题

### 1. Protobuf版本不兼容

如果遇到Protobuf版本问题，可以指定Protobuf路径：
```bash
cmake -DProtobuf_DIR=/path/to/protobuf/cmake ..
```

### 2. ZeroMQ找不到

如果CMake找不到ZeroMQ，可以手动指定：
```bash
cmake -DZMQ_INCLUDE_DIR=/usr/include -DZMQ_LIBRARY=/usr/lib/libzmq.so ..
```

### 3. 编译错误：找不到头文件

确保所有依赖库都已正确安装，并且CMake能够找到它们。可以检查CMake的输出信息。

## 开发模式

如果需要调试，可以使用Debug模式编译：
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## 清理

清理编译文件：
```bash
cd build
make clean
```

完全清理（包括CMake缓存）：
```bash
rm -rf build
```

