#
# $Id$
#

set(dir  data_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	SinglesRates 
	SinglesRatesForTimeFrames 
)
#ifeq ($(HAVE_LLN_MATRIX),1)
#  $(dir)_LIB_SOURCES += 
#	SinglesRatesFromSglFile 
#	SinglesRatesFromECAT7  
#endif

add_library(${dir} ${${dir_LIB_SOURCES}}    )

