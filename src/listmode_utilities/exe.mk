#
# $Id$
#

dir:=listmode_utilities

$(dir)_SOURCES = lm_to_projdata.cxx

ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile more

$(dir)_SOURCES += \
	scan_sgl_file.cxx \
	print_sgl_values.cxx \
	rebin_sgl_file.cxx 

endif

include $(WORKSPACE)/exe.mk
