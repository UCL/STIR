# Sample file (by Kris Thielemans)
# Use this file to modify defaults for some variables in Makefile_common
# Modify according to your own needs

# set non-default location of Louvain la Neuve files 
# This will (only) be used in utilities/Makefile to compile and link
# ifheaders_for_ecat7
LLN_INCLUDE_DIR=$(HOME)/lln/ecat

# This file only sets EXTRA_OPT
# Changes w.r.t. defaults:
# use all warnings,
# use debug flag -g even in optimised mode
# so, set EXTRA_OPT=-g -Wall
# for some systems, do more...
ifeq ($(SYSTEM),LINUX)
# our Linux system has Pentium III processors
EXTRA_OPT=-g -Wall -march=i686
else
ifeq ($(SYSTEM),CYGWIN)
# our NT system has Pentium II processors
# cygwin can't find X include files by default
EXTRA_OPT=-g -Wall -march=i686 -I /usr/X11R6/include
else
ifeq ($(SYSTEM),CC)
# our Parsytec CC system has gcc 2.8.1, so we don't use -Wall 
# and have to set the template depth
EXTRA_OPT= -ftemplate-depth-25
else
# default
EXTRA_OPT=-g -Wall
endif
endif
endif
