#
# $Id$
#

include ../Makefile_common


all: ${DEST}libdisplay.a 

clean:
	$(RM) $(DEST)*

#******* first find out which files we need

ifeq "$(GRAPHICS)" "X"
GRAPHOPT = -DTOMO_SIMPLE_BITMAPS -DSC_XWINDOWS
SOURCES_CXX = display_array.cxx 
SOURCES_C = gen.c screengen.c screen.c
else
ifeq "$(GRAPHICS)" "PGM"
GRAPHOPT = -DTOMO_PGM
SOURCES_CXX = display_array.cxx
SOURCES_C = 
else
ifeq "$(GRAPHICS)" "MATHLINK"
GRAPHOPT = -DTOMO_MATHLINK
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

include $(SOURCES_CXX:%.cxx=$(DEST)%.d) $(SOURCES_C:%.c=$(DEST)%.d)

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






