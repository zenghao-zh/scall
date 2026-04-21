foreach(target demo stress parse_dat perf)
    add_executable(${target} "${target}.cpp")

    target_include_directories(${target}
        PUBLIC
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            # spdlog from the original project
            ${PROJECT_SOURCE_DIR}/../ms_postprocessing/include
    )

    target_link_libraries(${target}
        PUBLIC
            ${PROJECT_NAME}
    )
endforeach()
