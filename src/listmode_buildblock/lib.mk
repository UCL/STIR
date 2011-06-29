#
# $Id$
#
dir := listmode_buildblock

$(dir)_LIB_SOURCES = \
	CListEvent.cxx \
	CListModeData.cxx \
	CListModeDataECAT.cxx \
	CListRecordECAT962.cxx \
	CListRecordECAT966.cxx \
	LmToProjData.cxx \
	LmToProjDataBootstrap.cxx \


#$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk
