#
# $Id$
#
# Author Kris Thielemans
#
# Copyright 2004- $Date$ Hammersmith Imanet Ltd
# This file is part of STIR.
#
#   This file is free software; you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation; either version 2.1 of the License, or
#   (at your option) any later version.
#
#   This file is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   See STIR/LICENSE.txt for details
#
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

# WORKSPACE macro should be set to the root of the source files
# it is now (largely?) irrelevant is make is only
# executed in 1 directory.
# WORKSPACE default to the current directory
# this can be done simply using . (as it's no longer a recursive make)
# Using . instead of $(CURDIR) avoids problems with spaces in filename
WORKSPACE := .
#$(CURDIR)

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
	numerics_buildblock \
	eval_buildblock Shape_buildblock \
	listmode_buildblock \
	iterative/LogLik_buildblock \
	iterative/OSMAPOSL \
	analytic/FBP2D \
	analytic/FBP3DRP

EXEDIRS += utilities recon_test \
	listmode_utilities \
	iterative/OSMAPOSL \
	iterative/sensitivity \
	analytic/FBP2D \
	analytic/FBP3DRP

TESTDIRS += test recon_test

STIR_LIB:=$(DEST)$(LIB_PREFIX)stir$(LIB_SUFFIX)

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
	rm -f $(STIR_LIB)

clean_exes:  $(addprefix clean_exes_, $(EXEDIRS)) 

clean: clean_tests clean_lib clean_exes
#	rm -rf $(DEST)

install: all install-exes

install-all: install-exes install-lib install-include

uninstall: uninstall-exes

uninstall-all: uninstall-exes uninstall-lib uninstall-include

run_tests: $(addprefix run_tests_, $(TESTDIRS)) 

run_interactive_tests: $(addprefix run_interactive_tests_, $(TESTDIRS)) 

# next target is necessary to get $(dir)_run_tests to compile/link all its files first
all_test_exes: $(foreach dir, $(TESTDIRS), $($(dir)_TEST_EXES))

.PRECIOUS:  $(DEST)*$(LIB_SUFFIX)

lib: $(STIR_LIB)

$(STIR_LIB): $(foreach dir, $(LIBDIRS), $($(dir)_LIB_OBJS))
ifneq ("$(AR)","link")
	$(AR) $(ARFLAGS)  $(AR_OUTFLAG)$@ $?
ifeq ("$(IS_GNU_AR)","")
	ranlib $@
endif
else
	$(AR) $(ARFLAGS)  $(AR_OUTFLAG)$@ $^
# link /lib seems to need a list of all .objs
endif



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
ifeq ("$(IS_MS_VC)","")
# normal case
MAKEDEPENDCXX = $(CXX) $(MAKE_DEPEND_FLAG) $(CFLAGS) > $(df).d $<
MAKEDEPENDC = $(CC) $(MAKE_DEPEND_FLAG) $(CFLAGS)  > $(df).d $<
else
# use when your compiler cannot generate dependencies in text form
# (Visual C++ can generate a database file, but we cannot read that)
# of course, this only works when you have makedepend, touch, sed

# for VC, the -Y option disables looking for system includes
# it wouldn't work for other compilers presumably
MAKEDEPENDCXX = touch $(df).d && makedepend $(CFLAGS) -f $(df).d $< -Y -o$(O_SUFFIX) 2> /dev/null
MAKEDEPENDC = $(MAKEDEPENDCXX)
endif

# note: first sed line put $(DEST) in front of the object target
# it uses s& to avoid problems with / in paths
# also, the \ in front of ${O_SUFFIX} is to 'escape' the . at the start of ${O_SUFFIX}
# as a . in the sed pattern means 'any character'
dotD2dotP = sed -e 's&$(*)\${O_SUFFIX}&$(DEST)$(*)$(O_SUFFIX)&' < $(df).d > $(df).P; \
	sed -e 's/\#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
                -e '/^$$/ d' -e 's/$$/ :/' < $(df).d >> $(df).P; \
	$(RM) $(df).d


${DEST}%${O_SUFFIX} : %.cxx
	@ -mkdir -p $(dir $@); 
	$(CXX) $(CFLAGS)  ${O_OUTFLAG}$@  -c $< 
ifeq ("$(DISABLE_DEPENDENCY_GENERATION)","")
	@ $(MAKEDEPENDCXX);$(dotD2dotP);
endif

${DEST}%${O_SUFFIX} : %.c
	@ -mkdir -p $(dir $@); 
	$(CC) $(CFLAGS) ${O_OUTFLAG}$@ -c $< 
ifeq ("$(DISABLE_DEPENDENCY_GENERATION)","")
	@ $(MAKEDEPENDC); $(dotD2dotP); 
endif	


endif # GCC3

# Default rule for making executables
# Note: this rule has to occur AFTER the definition of the macros
# occuring in the dependencies.
# Otherwise the value of the macros is empty when 
# checking the dependencies (even when they are alright when
# executing the corresponding command)
${DEST}%: ${DEST}%${O_SUFFIX} $(STIR_LIB) $(STIR_REGISTRIES)  $(EXTRA_LIBS) 
	${LINK} ${EXE_OUTFLAG}${@}$(EXE_SUFFIX) $<  $(STIR_REGISTRIES) $(STIR_LIB) $(LINKFLAGS) $(SYS_LIBS)
# DO NOT DELETE
