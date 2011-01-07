foreach(executable ${${dir_EXE_SOURCES}})
   add_executable(${executable} ${executable}.cxx ${STIR_REGISTRIES})
   target_link_libraries(${executable} ${STIR_LIBRARIES})
endforeach()

install(TARGETS ${${dir_EXE_SOURCES}} DESTINATION bin)
