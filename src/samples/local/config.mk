# $Id$
# Sample file for local/config.mk (by Kris Thielemans)
# See the STIR User's Guide.
# Use this file to modify defaults for some variables in the 'global' config.mk
# Modify according to your own needs
# i.e. DO NOT just copy it as it won't work for you.
#
# Note that this file is largely obsolete as usging CMake is now recommended.

# set non-default location of Louvain la Neuve files 
LLN_INCLUDE_DIR=$(HOME)/lln/ecat

# set non-default location where files will be installed
# this sets it to the "standard" location, but this means you need 
# superuser permissions
INSTALL_PREFIX=/usr/local

# This file only sets EXTRA_OPT
# Changes w.r.t. defaults:
# use all warnings,
# use debug flag -g even in optimised mode
# so, set EXTRA_OPT=-g -Wall
# for some systems, do more...
ifeq ($(SYSTEM),LINUX)
# always compile with debug, have all warnings on, 
# optimise for your current processor (compiled code might fail on other systems)
EXTRA_CFLAGS=-g -Wall -march=native
else
ifeq ($(SYSTEM),CYGWIN)
# an  incredibly old NT system has Pentium II processors
# maybe cygwin can't find X include files by default
EXTRA_CFLAGS=-g -Wall -march=pentium2 -I /usr/X11R6/include
else
ifeq ($(SYSTEM),CC)
# our Parsytec CC system had gcc 2.8.1, so we don't use -Wall 
# and have to set the template depth
EXTRA_CFLAGS= -ftemplate-depth-25
else
# default
EXTRA_CFLAGS=-g -Wall
endif
endif
endif
