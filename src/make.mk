#
# $Id$
#
WORKSPACE := $(CURDIR)

# make sure that 'all' is also the first target and hence the default
default_target: all


include config.mk

LIBDIRS :=
EXEDIRS :=
TESTDIRS:=
-include local/extra_dirs.mk

LIBDIRS += buildblock recon_buildblock display IO \
	eval_buildblock Shape_buildblock \
	listmode_buildblock \
	iterative/LogLik_buildblock \
	iterative/OSMAPOSL

EXEDIRS += utilities recon_test \
	listmode_utilities \
	iterative/OSMAPOSL \
	iterative/sensitivity

TESTDIRS += test recon_test

.PHONY: all clean lib install run_tests run_interactive_tests all_test_exes default_target


include $(addsuffix /lib.mk, $(LIBDIRS))
include $(addsuffix /exe.mk, $(EXEDIRS))
include $(addsuffix /test.mk, $(TESTDIRS))


all: $(EXEDIRS)

clean: $(addprefix clean_, $(EXEDIRS) $(LIBDIRS) $(TESTDIRS)) 
	rm -f $(DEST)libSTIR.a
#	rm -rf $(DEST)


install: $(addprefix install_, $(EXEDIRS))

run_tests: $(addprefix run_tests_, $(TESTDIRS)) 

run_interactive_tests: $(addprefix run_interactive_tests_, $(TESTDIRS)) 

# next target is necessary to get $(dir)_run_tests to compile all its files first
all_test_exes: $(foreach dir, $(TESTDIRS), $($(dir)_TEST_EXES))

.PRECIOUS:  $(DEST)*.a

STIR_LIB:=$(DEST)libstir.a

lib: $(STIR_LIB)


$(STIR_LIB): $(foreach dir, $(LIBDIRS), $($(dir)_LIB_OBJS))
	$(AR) $(ARFLAGS)  $@ $?
	ranlib $@




# STIR_REGISTRIES should be set to all object files we want to link with
# In particular, this should include all global variables of type 
# RegisteredParsingObject::RegisterIt
STIR_REGISTRIES:=$(foreach dir, $(LIBDIRS), $($(dir)_REGISTRY_OBJS))

#********* default rules

# These are the rules for compiling (linking rules are in Makefile_bblibs).

# There is some complicated trickery to get automatic dependency checking on
# .h and .inl files. This means that if you change only a .h file, make will 
# still rebuild all .c and .cxx files that include this .h file (even indirectly).

# Warning: if you change these, you should check display/Makefile where
# they are substituted by something appropriate for that directory.

.SUFFIXES: .c .o .cxx .a .P 

# To generate dependencies, we use -MM for gcc (or g++) and -M for other compilers
ifeq ($(CXX),g++)
MAKE_DEPEND_FLAG=-MM
else
ifeq ($(CC),gcc)
MAKE_DEPEND_FLAG=-MM
else
MAKE_DEPEND_FLAG=-M
endif
endif

# default rule for making 'mini-Makefiles' with the dependency info 
# for a single source file
# See http://make.paulandlesley.org/autodep.html

# first make a variable that allows us to test if we have gcc 3.0 or later
# If so, this will allow us to simplify life a lot!
IS_GCC_3:=$(shell $(CXX) -v 2>&1 |grep "gcc version [3456789]")

ifneq ("$(IS_GCC_3)","")

#$(warning Using rules for gcc 3 or later)

# use  gcc -MD -MP flags to obtain the .P files at the same time as the 
# ordinary compilation

${DEST}%.o : %.cxx
	@ -mkdir -p $(dir $@); 	
	$(CXX) $(CFLAGS) -o $@ ${MAKE_DEPEND_FLAG}D -MP -c $< 
	@ mv $(DEST)$(*).d $(DEST)$(*).P

${DEST}%.o : %.c
	@ -mkdir -p $(dir $@); 
	$(CC) $(CFLAGS) -o $@ ${MAKE_DEPEND_FLAG}D -MP  -c $< ;
	@ mv $(DEST)$(*).d $(DEST)$(*).P

else

#$(warning Using rules for non-gcc compilers (or gcc 2.*))

# we have to follow the original scheme of Paul D. Smith
# Modifications by KT:
# - handle DEST (in definition of df and by replacing the line "cp $(df.d) $(df.P)"
#   to a sed line that inserts $(DEST) before the name of the .o file
# - declare a variable dotD2dotP with all the sed stuff to create the .P file.
#   Note that to get this to work, I had to escape (i.e. put a backslash in front of)
#  the # sign in Paul's sed pattern. Otherwise, make interpretes it as the start
# of a comment.

df = $(DEST)$*

# alternative choices  for MAKEDEPEND
#MAKEDEPENDCXX = touch $*.d && makedepend $(CFLAGS) -f $(df).d $<
#MAKEDEPENDC = touch $*.d && makedepend $(CFLAGS) -f $(df).d $<
MAKEDEPENDCXX = $(CXX) $(MAKE_DEPEND_FLAG) $(CFLAGS) > $(df).d $<
MAKEDEPENDC = $(CC) $(MAKE_DEPEND_FLAG) $(CFLAGS)  > $(df).d $<

#cp $(df).d $(df).P; 
dotD2dotP = sed -e 's&$(*F)\.o&$(DEST)$(*).o&' < $(df).d > $(df).P; \
	sed -e 's/\#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
                -e '/^$$/ d' -e 's/$$/ :/' < $(df).d >> $(df).P; \
	rm -f $(df).d


${DEST}%.o : %.cxx
	@ -mkdir -p $(dir $@); \
	$(MAKEDEPENDCXX);  \
	$(dotD2dotP);
	$(CXX) $(CFLAGS) -o $@ -c $< 

${DEST}%.o : %.c
	@ -mkdir -p $(dir $@); \
	$(MAKEDEPENDC); \
	$(dotD2dotP); 
	$(CC) $(CFLAGS) -o $@ -c $< 


endif # GCC3

# Default rule for making executables
# Note: this rule has to occur AFTER the definition of the BB_LIBS
# et al macros otherwise the value of the macros is empty when 
# checking the dependencies (even when they are alright when
# executing the corresponding command)
${DEST}%: ${DEST}%.o $(STIR_LIB) $(STIR_REGISTRIES)
	$(CXX) $(CFLAGS)  -o $@ $< $(STIR_REGISTRIES) $(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)
