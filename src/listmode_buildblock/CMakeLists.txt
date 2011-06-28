#
# $Id$
#
set(dir listmode_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	CListEvent 
	CListModeData 
	LmToProjData 
)

if (HAVE_ECAT)
  list(APPEND ${dir_LIB_SOURCES}
	CListModeDataECAT 
	CListRecordECAT962 
	CListRecordECAT966 
  )
endif()

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries


add_library(${dir} ${${dir_LIB_SOURCES}}    )
