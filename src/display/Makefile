#
# $Id$
#
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common


all: ${DEST}libdisplay.a 

clean:
	$(RM) $(DEST)*

#******* first find out which files we need

ifeq "$(GRAPHICS)" "X"
GRAPHOPT = -DSTIR_SIMPLE_BITMAPS -DSC_XWINDOWS
SOURCES_CXX = display_array.cxx 
SOURCES_C = gen.c screengen.c screen.c
else
ifeq "$(GRAPHICS)" "PGM"
GRAPHOPT = -DSTIR_PGM
SOURCES_CXX = display_array.cxx
SOURCES_C = 
else
ifeq "$(GRAPHICS)" "MATHLINK"
GRAPHOPT = -DSTIR_MATHLINK
SOURCES_CXX = display_array.cxx 
SOURCES_C = mathlinkhelp.c
else
GRAPHOPT =
SOURCES_CXX= display_array.cxx
SOURCES_C = 
endif
endif
endif


#******* now set dependencies

OBJS = $(SOURCES_CXX:%.cxx=$(DEST)%.o) $(SOURCES_C:%.c=$(DEST)%.o)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES_CXX:%.cxx=$(DEST)%.d) $(SOURCES_C:%.c=$(DEST)%.d)
endif


.PRECIOUS: $(DEST)*.o $(DEST)*.a

${DEST}libdisplay.a:  $(OBJS)
	ar r $@ $?
	ranlib $@


#******* default rules
# disable default rules from Makefile_common

${DEST}%.o: %.cxx 

${DEST}%.o: %.c


# new rules use GRAPHOPT

${DEST}%.o: %.cxx 
	-mkdir -p $(DEST)
	$(CXX) $(CFLAGS) -o $@ -c $< $(GRAPHOPT)

${DEST}%.o: %.c
	-mkdir -p $(DEST)
	$(CC) $(CFLAGS) -o $@ -c $< $(GRAPHOPT)






