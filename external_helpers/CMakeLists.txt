# include all git submodules that should be built with STIR
add_subdirectory(fmt)

# override default compile variables of fmt to make it compatible with shared objects
if (TARGET fmt)
    target_compile_options(fmt PRIVATE "-fPIC")
endif()