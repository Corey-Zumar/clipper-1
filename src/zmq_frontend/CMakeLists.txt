cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_executable(clipper_zmq_frontend
        src/zmq_frontend_main.cpp
        src/frontend_rpc_service.cpp)

target_link_libraries(clipper_zmq_frontend
        clipper
        boost
        cxxopts
        httpserver
        zmqcpp
        base64json
)
