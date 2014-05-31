#
#
dir := listmode_buildblock

$(dir)_LIB_SOURCES = \
	CListEvent.cxx \
	CListModeData.cxx \
	LmToProjData.cxx \
	LmToProjDataBootstrap.cxx \
	CListModeDataECAT8_32bit.cxx \
	CListRecordECAT8_32bit.cxx

ifeq ($(HAVE_LLN_MATRIX),1)
  $(dir)_LIB_SOURCES +=  \
	CListModeDataECAT.cxx \
	CListRecordECAT962.cxx \
	CListRecordECAT966.cxx
endif

#$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk
