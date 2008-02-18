#
# $Id$
#

dir := local/IO

$(dir)_LIB_SOURCES:= \
	InterfileDynamicDiscretisedDensityOutputFileFormat.cxx \
	InterfileParametricDensityOutputFileFormat.cxx \
	OutputFileFormat_default.cxx

ifeq ($(HAVE_LLN_MATRIX),1)
  $(dir)_LIB_SOURCES += ECAT7ParametricDensityOutputFileFormat.cxx \
		  ECAT7DynamicDiscretisedDensityOutputFileFormat.cxx
endif



#$(dir)_REGISTRY_SOURCES:= local_IO_registries.cxx

include $(WORKSPACE)/lib.mk

