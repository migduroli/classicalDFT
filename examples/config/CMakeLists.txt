add_executable(example_config main.cpp)
target_link_libraries(example_config PRIVATE classicaldft)

configure_file(config_files/config.ini  ${CMAKE_CURRENT_BINARY_DIR}/config.ini  COPYONLY)
configure_file(config_files/config.json ${CMAKE_CURRENT_BINARY_DIR}/config.json COPYONLY)
configure_file(config_files/config.xml  ${CMAKE_CURRENT_BINARY_DIR}/config.xml  COPYONLY)
configure_file(config_files/config.info ${CMAKE_CURRENT_BINARY_DIR}/config.info COPYONLY)
