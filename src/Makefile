#
# $Id$
#
# Author Kris Thielemans
# Copyright 2004- $Date$ Hammersmith Imanet Ltd
# This file is part of STIR, and is distributed under the 
# terms of the GNU Lesser General Public Licence (LGPL) Version 2.1.
#
# This makefile defines dependencies and rules for the 
# following targets:
#	how (gives brief usage info)	
#	all clean lib install install-all uninstall uninstall-all
#	clean_tests clean_lib clean_exes 
#	run_tests run_interactive_tests 
#
# It works by using the 'recursive makefile' scheme.
# It is set-up to recurse into $(LIBDIRS), $(EXEDIRS), $(TESTDIRS)
# from where it imports lib.mk, exe.mk and test.mk respectively.
# These  'sub'makefiles will define other targets such as 
# EXESUBDIR, clean_exes_EXESUBDIR, install_exes_EXESUBDIR
# build_tests_TESTSUBDIR, clean_tests_TESTSUBDIR, 
#                   run_tests_TESTSUBDIR, 
#                   run_interactive_tests_TESTSUBDIR
# build_lib_LIBSUBDIR, clean_lib_LIBSUBDIR
# See lib.mk, exe.mk and test.mk in this directory for more details.
#
.PHONY: how all clean lib \
	install install-all install-exes install-lib install-include \
	uninstall uninstall-all uninstall-exes uninstall-lib uninstall-include \
	clean_tests clean_lib clean_exes \
	run_tests run_interactive_tests all_test_exes \
	default_target

WORKSPACE := $(CURDIR)

# make sure that 'all' is also the first target and hence the default
default_target: all

how:
	@echo "Usage: make [target] [options]" 
	@echo "where useful targets are:"
	@echo "     all clean lib install install-all"
	@echo "     uninstall uninstall-all"
	@echo "     clean_tests clean_lib clean_exes "
	@echo "     run_tests run_interactive_tests" 
	@echo "and options can be for instance:"
	@echo "     BUILD=debug DEST=somenicedir/ INSTALL_PREFIX=/usr/local"
	@echo "See the STIR User's guide for more info."


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

STIR_LIB:=$(DEST)libstir.a

include $(addsuffix /lib.mk, $(LIBDIRS))


# STIR_REGISTRIES should be set to all object files we want to 
# explicitly link with (i.e. in addition to the library).
# In particular, this should include all files with
# global variables of type 
# RegisteredParsingObject::RegisterIt
# We set it here such that the sub-Makefiles below can use it
STIR_REGISTRIES:=$(foreach dir, $(LIBDIRS), $($(dir)_REGISTRY_OBJS))

include $(addsuffix /exe.mk, $(EXEDIRS))
include $(addsuffix /test.mk, $(TESTDIRS))


all: $(EXEDIRS)

clean_tests: $(addprefix clean_tests_, $(TESTDIRS)) 

clean_lib:  $(addprefix clean_lib_, $(LIBDIRS)) 
	rm -f $(DEST)libSTIR.a

clean_exes:  $(addprefix clean_exes_, $(EXEDIRS)) 

clean: clean_tests clean_lib clean_exes
#	rm -rf $(DEST)

install: install-exes

install-all: install-exes install-lib install-include

uninstall: uninstall-exes

uninstall-all: uninstall-exes uninstall-lib uninstall-include

run_tests: $(addprefix run_tests_, $(TESTDIRS)) 

run_interactive_tests: $(addprefix run_interactive_tests_, $(TESTDIRS)) 

# next target is necessary to get $(dir)_run_tests to compile/link all its files first
all_test_exes: $(foreach dir, $(TESTDIRS), $($(dir)_TEST_EXES))

.PRECIOUS:  $(DEST)*.a

lib: $(STIR_LIB)


$(STIR_LIB): $(foreach dir, $(LIBDIRS), $($(dir)_LIB_OBJS))
	$(AR) $(ARFLAGS)  $@ $?
	ranlib $@




install-exes: $(addprefix install_exes_, $(EXEDIRS))


uninstall-exes: $(addprefix uninstall_exes_, $(EXEDIRS))

install-lib: lib $(STIR_REGISTRIES)
	mkdir -p $(INSTALL_LIB_DIR)
	$(INSTALL) $(STIR_LIB) $(STIR_REGISTRIES) $(INSTALL_LIB_DIR)

uninstall-lib: 
	$(RM) $(addprefix $(INSTALL_LIB_DIR)/, $(notdir $(STIR_LIB) $(STIR_REGISTRIES)))

install-include:
	mkdir -p $(INSTALL_INCLUDE_DIR)
	cp -R ${WORKSPACE}/include/stir $(INSTALL_INCLUDE_DIR)
	@echo Check permissions for the files in $(INSTALL_INCLUDE_DIR)/stir
	@echo Warning: boost include files are not copied

uninstall-include:
	$(RM) -r $(INSTALL_INCLUDE_DIR)/stir

#********* default rules

# These are the rules for compiling and linking.

# There is some complicated trickery to get automatic dependency checking on
# .h and .inl files. This means that if you change only a .h file, make will 
# still rebuild all .c and .cxx files that include this .h file (even indirectly).

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
# Note: this rule has to occur AFTER the definition of the macros
# occuring in the dependencies.
# Otherwise the value of the macros is empty when 
# checking the dependencies (even when they are alright when
# executing the corresponding command)
${DEST}%: ${DEST}%.o $(STIR_LIB) $(STIR_REGISTRIES)
	$(CXX) $(CFLAGS)  -o $@ $< $(STIR_REGISTRIES) $(STIR_LIB)  $(LINK_OPT) $(SYS_LIBS)
