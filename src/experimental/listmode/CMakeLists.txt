#
#
set(dir local_listmode_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

)

if (HAVE_ECAT)
#  list(APPEND ${dir_LIB_SOURCES}
#  )
endif()

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries


include(stir_lib_target)

target_link_libraries(local_listmode_buildblock PUBLIC listmode_buildblock  )
