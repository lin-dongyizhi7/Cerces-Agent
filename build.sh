#!/bin/bash
# 构建脚本 - Agent 和 Communicator

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo -e "${GREEN}=== Cerces-Agent 构建脚本 ===${NC}"

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}错误: 未找到 CMake${NC}"
    echo "请安装: brew install cmake (macOS) 或 apt-get install cmake (Linux)"
    exit 1
fi
echo -e "${GREEN}✓ CMake: $(cmake --version | head -n1)${NC}"

# 检查 Protobuf
if ! command -v protoc &> /dev/null; then
    echo -e "${RED}错误: 未找到 Protobuf 编译器${NC}"
    echo "请安装: brew install protobuf (macOS) 或 apt-get install protobuf-compiler (Linux)"
    exit 1
fi
echo -e "${GREEN}✓ Protobuf: $(protoc --version)${NC}"

# 检查 ZeroMQ
if command -v pkg-config &> /dev/null; then
    if pkg-config --exists libzmq; then
        echo -e "${GREEN}✓ ZeroMQ: $(pkg-config --modversion libzmq)${NC}"
    else
        echo -e "${YELLOW}警告: 未找到 ZeroMQ，构建可能失败${NC}"
        echo "请安装: brew install zeromq (macOS) 或 apt-get install libzmq3-dev (Linux)"
    fi
else
    echo -e "${YELLOW}警告: 未找到 pkg-config，无法检查 ZeroMQ${NC}"
fi

# 检查编译器
if command -v g++ &> /dev/null; then
    echo -e "${GREEN}✓ 编译器: $(g++ --version | head -n1)${NC}"
elif command -v clang++ &> /dev/null; then
    echo -e "${GREEN}✓ 编译器: $(clang++ --version | head -n1)${NC}"
else
    echo -e "${RED}错误: 未找到 C++ 编译器${NC}"
    exit 1
fi

# 解析命令行参数
BUILD_TYPE="Release"
INSTALL_PREFIX=""
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --debug       Debug 模式构建"
            echo "  --release     Release 模式构建（默认）"
            echo "  --prefix PATH 指定安装路径"
            echo "  --clean       清理构建目录"
            echo "  -h, --help    显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            exit 1
            ;;
    esac
done

# 清理构建目录
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}清理构建目录...${NC}"
    rm -rf "$BUILD_DIR"
fi

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 运行 CMake
echo -e "${YELLOW}运行 CMake 配置...${NC}"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX")
fi

# macOS 特定设置
if [[ "$OSTYPE" == "darwin"* ]]; then
    # 尝试使用 Homebrew 路径
    if [ -d "/opt/homebrew" ]; then
        export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
        export CMAKE_PREFIX_PATH="/opt/homebrew:$CMAKE_PREFIX_PATH"
    elif [ -d "/usr/local" ]; then
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
        export CMAKE_PREFIX_PATH="/usr/local:$CMAKE_PREFIX_PATH"
    fi
fi

cmake "${CMAKE_ARGS[@]}" "$PROJECT_ROOT"

# 编译
echo -e "${YELLOW}开始编译...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc)
fi

make -j"$CORES"

# 检查构建结果
echo -e "${YELLOW}验证构建结果...${NC}"
if [ -f "agent" ] && [ -f "communicator" ]; then
    echo -e "${GREEN}✓ 构建成功！${NC}"
    echo ""
    echo "生成的可执行文件:"
    ls -lh agent communicator
    echo ""
    echo "运行方式:"
    echo "  ./communicator ../config/communicator.conf"
    echo "  ./agent localhost 8888 node_0 rank_0 1000"
else
    echo -e "${RED}✗ 构建失败：未找到可执行文件${NC}"
    exit 1
fi

