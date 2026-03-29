# ── Modern cdft examples ─────────────────────────────────────────────────────

add_executable(cdft_eos eos.cpp)
target_link_libraries(cdft_eos PRIVATE classicaldft)

add_executable(cdft_potentials potentials.cpp)
target_link_libraries(cdft_potentials PRIVATE classicaldft)

add_executable(cdft_fmt fmt.cpp)
target_link_libraries(cdft_fmt PRIVATE classicaldft)

add_executable(cdft_crystal crystal.cpp)
target_link_libraries(cdft_crystal PRIVATE classicaldft)

add_executable(cdft_numerics numerics.cpp)
target_link_libraries(cdft_numerics PRIVATE classicaldft)

add_executable(cdft_config config.cpp)
target_link_libraries(cdft_config PRIVATE classicaldft)
