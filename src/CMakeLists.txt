# Create the Python module
pybind11_add_module(backtester_core MODULE
    backtester.cpp
    python_bindings.cpp
)

# Set include directories for the target
target_include_directories(backtester_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(backtester_core
    PRIVATE
        fmt::fmt
        nlohmann_json::nlohmann_json
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        ${Python3_LIBRARIES}
)

# Set compile options for macOS
if(APPLE)
    target_compile_options(backtester_core 
        PRIVATE
            -Wall
            -Wextra
            -stdlib=libc++
            -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    )
    
    target_link_options(backtester_core
        PRIVATE
            -stdlib=libc++
    )
endif()

# Set output properties
set_target_properties(backtester_core 
    PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
)
