#
# $Id$
#

dir := local/IO

$(dir)_LIB_SOURCES:= \
	local_OutputFileFormat_default.cxx \
	local_InputFileFormatRegistry.cxx




#$(dir)_REGISTRY_SOURCES:= local_IO_registries.cxx

include $(WORKSPACE)/lib.mk

