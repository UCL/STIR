#
# $Id$
#

dir:=utilities/ecat

$(dir)_SOURCES:=""
ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile 
  # ifheaders_for_ecat7 etc as well
  $(dir)_SOURCES += is_ecat7_file.cxx copy_ecat7_header.cxx \
	ifheaders_for_ecat7.cxx conv_to_ecat7.cxx print_ecat_singles_values.cxx \
	convecat6_if.cxx \
        conv_to_ecat6.cxx \
	ecat_swap_corners.cxx 
endif

${DEST}$(dir)/is_ecat7_file: ${DEST}$(dir)/is_ecat7_file$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)

${DEST}$(dir)/copy_ecat7_header: ${DEST}$(dir)/copy_ecat7_header$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)


include $(WORKSPACE)/exe.mk
