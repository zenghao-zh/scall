set(TARGET ${PROJECT_NAME})

file(GLOB_RECURSE srcs
    rta_postprocess.cpp
    postprocess.cpp
    kernel_cpu.cpp
)

add_library(${TARGET} SHARED ${srcs})

target_include_directories(${TARGET}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        # reuse spdlog headers from the original project
        ${PROJECT_SOURCE_DIR}/../ms_postprocessing/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PUBLIC OpenMP::OpenMP_CXX)
endif()

SET_TARGET_PROPERTIES(${TARGET} PROPERTIES
    VERSION ${MSPOSTPROCESS_VERSION_MAJOR}.${MSPOSTPROCESS_VERSION_MINOR}.${MSPOSTPROCESS_VERSION_REV}
)
