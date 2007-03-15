#
# $Id$
#

dir:=utilities/ecat

$(dir)_SOURCES:=""
ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile 
  # ifheaders_for_ecat7 etc as well
  $(dir)_SOURCES += is_ecat7_file.cxx
endif

${DEST}$(dir)/is_ecat7_file: ${DEST}$(dir)/is_ecat7_file$(O_SUFFIX) \
   $(STIR_LIB) 
	$(LINK) $(EXE_OUTFLAG)$@ $< \
		 $(STIR_LIB)  $(LINKFLAGS) $(SYS_LIBS)



include $(WORKSPACE)/exe.mk
