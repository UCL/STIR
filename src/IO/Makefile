#
# $Id$
#
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common


all: $(DEST)libIO.a $(DEST)IO_registries.o

clean:
	$(RM) $(DEST)*.[oadP]

SOURCES = \
  OutputFileFormat.cxx \
  InterfileOutputFileFormat.cxx \
  interfile.cxx InterfileHeader.cxx \
  ECAT6OutputFileFormat.cxx \
  stir_ecat_common.cxx \
  stir_ecat6.cxx ecat6_utils.cxx
 

ifeq ($(HAVE_LLN_MATRIX),1)
  SOURCES += ECAT7OutputFileFormat.cxx stir_ecat7.cxx
endif


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)
#LOBJS = $(DEST)libIO.a($(OBJS))

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif

.PRECIOUS:  $(DEST)*.a $(DEST)*.o



$(DEST)libIO.a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $?
	ranlib $@


