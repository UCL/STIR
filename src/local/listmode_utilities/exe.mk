#
# $Id$
#

dir:=local/listmode_utilities

$(dir)_SOURCES = \
	lm_to_projdata_bootstrap.cxx \
	lm_fansums.cxx \
	generate_headcurve.cxx \
	list_time_events.cxx \
	change_lm_time_tags.cxx 

ifeq ($(STIR_DEVEL_MOTION),1)
$(dir)_SOURCES+= \
	lm_to_projdata_with_MC.cxx 
endif

ifeq ($(HAVE_LLN_MATRIX),1)
  $(dir)_SOURCES += \
	get_singles_info.cxx
endif

include $(WORKSPACE)/exe.mk
