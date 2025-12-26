cmake_minimum_required(VERSION 3.10)
project(CercesAgent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# 使用 pkg-config 获取完整的 Protobuf 链接库（包含 Abseil）
pkg_check_modules(PROTOBUF_PKG REQUIRED protobuf)

# 优先使用 homebrew 的 protoc，确保版本与运行时库一致
# 首先尝试从 pkg-config 获取 protoc 路径
execute_process(
    COMMAND pkg-config --variable=prefix protobuf
    OUTPUT_VARIABLE PROTOBUF_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# 查找 protoc，优先使用 homebrew 路径
find_program(Protobuf_PROTOC_EXECUTABLE
    NAMES protoc
    PATHS
        ${PROTOBUF_PREFIX}/bin
        /opt/homebrew/bin
        /usr/local/bin
        /usr/bin
    NO_DEFAULT_PATH
)

# 如果还没找到，尝试在 PATH 中查找（但排除 conda）
if(NOT Protobuf_PROTOC_EXECUTABLE)
    find_program(Protobuf_PROTOC_EXECUTABLE
        NAMES protoc
        PATHS
            /opt/homebrew/bin
            /usr/local/bin
            /usr/bin
    )
endif()

# 如果还是没找到，使用 find_package
if(NOT Protobuf_PROTOC_EXECUTABLE)
    find_package(Protobuf REQUIRED)
endif()

# 确保使用正确的 protoc
if(NOT Protobuf_PROTOC_EXECUTABLE)
    message(FATAL_ERROR "protoc not found. Please ensure protobuf is installed via homebrew.")
endif()

# 验证 protoc 版本
message(STATUS "Using protoc: ${Protobuf_PROTOC_EXECUTABLE}")
execute_process(
    COMMAND ${Protobuf_PROTOC_EXECUTABLE} --version
    OUTPUT_VARIABLE PROTOBUF_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "protoc version: ${PROTOBUF_VERSION_OUTPUT}")

# 检查版本是否匹配
if(PROTOBUF_VERSION_OUTPUT MATCHES "29\\.|5\\.29")
    message(WARNING "Detected old protoc version. This may cause compatibility issues.")
    message(WARNING "Please use homebrew protoc (version 33.x) instead of conda protoc.")
endif()

# 查找 cppzmq (ZeroMQ C++ binding, header-only)
# 尝试查找 cppzmq 包
find_path(CPPZMQ_INCLUDE_DIR
    NAMES zmq.hpp
    PATHS
        /opt/homebrew/include
        /usr/local/include
        /usr/include
        ${ZMQ_INCLUDE_DIRS}
    PATH_SUFFIXES
        cppzmq
)

if(CPPZMQ_INCLUDE_DIR)
    message(STATUS "Found cppzmq: ${CPPZMQ_INCLUDE_DIR}")
else()
    message(WARNING "cppzmq not found. Please install: brew install cppzmq (macOS) or apt-get install libcppzmq-dev (Linux)")
    # 尝试使用 ZMQ_INCLUDE_DIRS 作为备选
    set(CPPZMQ_INCLUDE_DIR ${ZMQ_INCLUDE_DIRS})
endif()

# Protobuf生成
set(PROTO_FILES
    proto/monitor.proto
)

set(PROTO_SRCS)
set(PROTO_HDRS)

foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    get_filename_component(PROTO_PATH ${PROTO_FILE} PATH)
    
    set(PROTO_SRC "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_PATH}/${PROTO_NAME}.pb.cc")
    set(PROTO_HDR "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_PATH}/${PROTO_NAME}.pb.h")
    
    list(APPEND PROTO_SRCS ${PROTO_SRC})
    list(APPEND PROTO_HDRS ${PROTO_HDR})
    
    add_custom_command(
        OUTPUT ${PROTO_SRC} ${PROTO_HDR}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
             --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
             ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
        COMMENT "Generating ${PROTO_NAME}.pb.cc and ${PROTO_NAME}.pb.h"
    )
endforeach()

# Communicator可执行文件
add_executable(communicator
    Communicator/src/main.cpp
    Communicator/src/NetworkManager.cpp
    Communicator/src/ProtocolHandler.cpp
    Communicator/src/DataTransformer.cpp
    Communicator/src/MessageQueue.cpp
    Communicator/src/ConfigManager.cpp
    Communicator/src/PythonInterface.cpp
    ${PROTO_SRCS}
)

target_include_directories(communicator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Communicator/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/proto
    ${PROTOBUF_PKG_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
    ${CPPZMQ_INCLUDE_DIR}
)

target_link_libraries(communicator
    ${PROTOBUF_PKG_LIBRARIES}
    ${ZMQ_LIBRARIES}
    pthread
)

# 添加库路径（macOS Homebrew）
if(APPLE)
    target_link_directories(communicator PRIVATE
        ${PROTOBUF_PKG_LIBRARY_DIRS}
        ${ZMQ_LIBDIR}
        /opt/homebrew/lib
        /usr/local/lib
    )
endif()

# Agent可执行文件
add_executable(agent
    Agent/src/main.cpp
    Agent/src/MetricGenerator.cpp
    Agent/src/AgentClient.cpp
    ${PROTO_SRCS}
)

target_include_directories(agent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Agent/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/proto
    ${PROTOBUF_PKG_INCLUDE_DIRS}
)

target_link_libraries(agent
    ${PROTOBUF_PKG_LIBRARIES}
    pthread
)

# 添加 Protobuf 库路径（macOS Homebrew）
if(APPLE)
    target_link_directories(agent PRIVATE
        ${PROTOBUF_PKG_LIBRARY_DIRS}
        /opt/homebrew/lib
        /usr/local/lib
    )
endif()

# 安装规则
install(TARGETS communicator agent
    RUNTIME DESTINATION bin
)

# 复制配置文件
install(FILES config/communicator.conf
    DESTINATION etc
)

