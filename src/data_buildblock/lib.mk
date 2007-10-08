#
# $Id$
#

dir := data_buildblock
$(dir)_LIB_SOURCES:= \
	SinglesRates.cxx \
	SinglesRatesForTimeFrames.cxx 

ifeq ($(HAVE_LLN_MATRIX),1)
  $(dir)_LIB_SOURCES += \
	SinglesRatesFromSglFile.cxx \
	SinglesRatesFromECAT7.cxx  
endif

$(dir)_REGISTRY_SOURCES:= data_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk

