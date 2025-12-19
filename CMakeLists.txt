cmake_minimum_required(VERSION 3.10)
project(CercesAgent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

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
    ${Protobuf_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(communicator
    ${Protobuf_LIBRARIES}
    ${ZMQ_LIBRARIES}
    pthread
)

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
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(agent
    ${Protobuf_LIBRARIES}
    pthread
)

# 安装规则
install(TARGETS communicator agent
    RUNTIME DESTINATION bin
)

# 复制配置文件
install(FILES config/communicator.conf
    DESTINATION etc
)

