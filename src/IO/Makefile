#
# $Id$
#
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common


all: $(DEST)libIO.a $(DEST)IO_registries.o

clean:
	$(RM) $(DEST)*.[oad]

SOURCES = \
  OutputFileFormat.cxx \
  InterfileOutputFileFormat.cxx
 




OBJS = $(SOURCES:%.cxx=$(DEST)%.o)
LOBJS = $(DEST)libIO.a($(OBJS))

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


.PRECIOUS:  $(DEST)*.a $(DEST)*.o



$(DEST)libIO.a: $(LOBJS)
	ranlib $@
#	ar r $@ $?


