# 构建指南 - Agent 和 Communicator

## 一、依赖安装

### macOS 系统

```bash
# 安装 CMake
brew install cmake

# 安装 Protobuf（如果未安装）
brew install protobuf

# 安装 ZeroMQ
brew install zeromq

# 安装 pkg-config（用于查找库）
brew install pkg-config
```

### Ubuntu/Debian 系统

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    libprotobuf-dev \
    protobuf-compiler \
    libzmq3-dev \
    pkg-config
```

### CentOS/RHEL 系统

```bash
sudo yum install -y \
    cmake \
    gcc-c++ \
    protobuf-devel \
    protobuf-compiler \
    zeromq-devel \
    pkgconfig
```

## 二、构建步骤

### 方法 1：标准构建流程

```bash
# 1. 进入项目根目录
cd /Users/dp/Downloads/use/Cerces-Agent

# 2. 创建构建目录
mkdir -p build
cd build

# 3. 运行 CMake 配置
cmake ..

# 4. 编译（会同时构建 agent 和 communicator）
make -j$(nproc)  # Linux
# 或
make -j$(sysctl -n hw.ncpu)  # macOS

# 5. 验证构建结果
ls -lh agent communicator
```

### 方法 2：指定安装路径

```bash
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install  # 可选：安装到系统路径
```

### 方法 3：Debug 模式构建

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### 方法 4：Release 模式构建（优化）

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## 三、构建输出

构建成功后，在 `build/` 目录下会生成：

- **`agent`** - Agent 采集客户端可执行文件
- **`communicator`** - Communicator 通信层服务可执行文件

## 四、运行测试

### 1. 启动 Communicator

```bash
cd build
./communicator ../config/communicator.conf
```

### 2. 启动 Agent（在另一个终端）

```bash
cd build
./agent localhost 8888 node_0 rank_0 1000
```

参数说明：
- `localhost` - Communicator 服务地址
- `8888` - 端口号
- `node_0` - 节点名称
- `rank_0` - 进程 rank
- `1000` - 采集间隔（毫秒）

## 五、常见问题解决

### 1. CMake 找不到 Protobuf

```bash
# 手动指定 Protobuf 路径
cmake -DProtobuf_DIR=/path/to/protobuf/cmake ..
```

### 2. CMake 找不到 ZeroMQ

```bash
# 手动指定 ZeroMQ 路径
cmake \
    -DZMQ_INCLUDE_DIR=/usr/local/include \
    -DZMQ_LIBRARY=/usr/local/lib/libzmq.dylib \
    ..
```

### 3. macOS 上找不到库文件

如果使用 Homebrew 安装的库，可能需要设置环境变量：

```bash
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="/usr/local:$CMAKE_PREFIX_PATH"
cmake ..
```

### 4. 编译错误：C++17 特性不支持

确保使用支持 C++17 的编译器：

```bash
# 检查编译器版本
g++ --version  # 需要 7+
clang++ --version  # 需要 5+

# 如果版本过低，更新编译器或使用 conda 环境
```

### 5. 链接错误：找不到 pthread

这是正常的，CMakeLists.txt 中已包含 pthread 链接。

## 六、清理构建

### 清理编译文件（保留 CMake 配置）

```bash
cd build
make clean
```

### 完全清理（删除整个 build 目录）

```bash
rm -rf build
```

## 七、项目结构说明

根据 CMakeLists.txt，构建过程会：

1. **生成 Protobuf 代码**：从 `proto/monitor.proto` 生成 C++ 代码
2. **构建 Communicator**：
   - 源文件：`Communicator/src/*.cpp`
   - 头文件：`Communicator/include/*.h`
   - 依赖：Protobuf、ZeroMQ、pthread
3. **构建 Agent**：
   - 源文件：`Agent/src/*.cpp`
   - 头文件：`Agent/include/*.h`
   - 依赖：Protobuf、pthread

## 八、验证依赖

构建前可以运行以下命令验证依赖：

```bash
# 检查 CMake
cmake --version  # 需要 >= 3.10

# 检查 Protobuf
protoc --version

# 检查 ZeroMQ
pkg-config --modversion libzmq

# 检查编译器
g++ --version
```

