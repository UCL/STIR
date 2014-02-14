#
#

dir:=local/listmode_utilities

$(dir)_SOURCES = \
	generate_headcurve.cxx \
	lm_to_projdata_with_random_rejection.cxx

ifeq ($(STIR_DEVEL_966_FIX),1)
$(dir)_SOURCES+= \
	change_lm_time_tags.cxx 
endif

ifeq ($(STIR_DEVEL_MOTION),1)
$(dir)_SOURCES+= \
	lm_to_projdata_with_MC.cxx 
endif

ifeq ($(HAVE_LLN_MATRIX),1)
  $(dir)_SOURCES += \
	get_singles_info.cxx
endif

include $(WORKSPACE)/exe.mk
