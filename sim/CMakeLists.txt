add_library(tiny_moe_iss router_iss.cpp)
target_include_directories(tiny_moe_iss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(tiny_moe_iss_test router_iss_test.cpp)
target_link_libraries(tiny_moe_iss_test PRIVATE tiny_moe_iss)

enable_testing()
add_test(NAME tiny_moe_iss_test COMMAND tiny_moe_iss_test)
