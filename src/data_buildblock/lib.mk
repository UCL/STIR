#
# $Id$
#

dir := data_buildblock
$(dir)_LIB_SOURCES:= \
	SinglesRates.cxx \
	SinglesRatesFromSglFile.cxx \
	SinglesRatesFromECAT7.cxx  

$(dir)_REGISTRY_SOURCES:= data_buildblock_registries.cxx

include $(WORKSPACE)/lib.mk

