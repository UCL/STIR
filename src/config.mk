# 
# $Id$
#
# Author Kris Thielemans
# Copyright 2004- $Date$ Hammersmith Imanet Ltd
# This file is part of STIR, and is distributed under the 
# terms of the GNU Lesser General Public Licence (LGPL) Version 2.1.

#******* type of build, if BUILD is not 'debug', we make the optimised version
BUILD=opt

#******** location of files

INCLUDE_DIR = ${WORKSPACE}/include
INSTALL_PREFIX := ${HOME}
INSTALL_EXE_DIR = $(INSTALL_PREFIX)/bin
INSTALL_LIB_DIR = $(INSTALL_PREFIX)/lib
INSTALL_INCLUDE_DIR = $(INSTALL_PREFIX)/include

# location of .o, .a and executables, set DEST if you don't agree with the default
ifndef DEST
ifeq ($(BUILD),debug)
DEST=debug/
else # not debug
DEST=opt/
endif # debug/ ?
endif # DEST not defined

#******* type of graphics used by display()
# possible values: X, PGM, MATHLINK, NONE
# note: only used in display/lib.mk and for GRAPH_LIBS
GRAPHICS=X

#******* command used for install* targets
INSTALL=cp
# could be something like 
#INSTALL="install -s -m 755"

#****** find out which system we're on

include $(WORKSPACE)/Makefile_common_system_type

#******** type of parallel library
# PARALIB can be EPX or PVM
ifeq ($(SYSTEM),CC)
PARALIB=EPX
else
PARALIB=PVM
endif # not CC

#******** default compiler and linker that will be used

# for C++ 
CXX=g++ 
# for C
CC=gcc
# program that will be used for linking, normally the C++ compiler
LINK=$(CXX)


# make a variable that allows us to test if we have Microsoft Visual Studio C++
# If so, lots of variables in the makefiles will have to be set non-standard.
IS_MS_VC:=$(shell $(CXX) 2>&1  |grep Microsoft)

ifneq ("$(IS_MS_VC)","")
#$(warning Enabling Visual C++ specific fixes)
endif

#******* compiler and linker extra options


#** EXTRA_CFLAGS: for compiler
# allow the user to get some extra options by using make EXTRA_CFLAGS=bla 
ifeq ($(CC),gcc)
EXTRA_CFLAGS =-Wall -Wno-deprecated
endif

#** EXTRA_LINKFLAGS: for linker
#allow the user to get extra options for link time
EXTRA_LINKFLAGS=
#** compiler options


#******** variables used only for ecat7
# local/config.mk can override these defaults
LLN_INCLUDE_DIR=$(WORKSPACE)/../lln/ecat
LLN_LIB_DIR=$(LLN_INCLUDE_DIR)



#******* compiler and linker options

ifeq ($(SYSTEM),LINUX)
# note for gcc 2.95.2:
# do not use -malign-double as it crashes in iostream stuff
OPTIM_CFLAGS=-O3  -ffast-math -DNDEBUG
else
ifeq ($(SYSTEM),CYGWIN)
OPTIM_CFLAGS=-O3  -ffast-math -malign-double -DNDEBUG
else
OPTIM_CFLAGS=-O3 -DNDEBUG
endif
endif

DEBUG_CFLAGS=-D_DEBUG -g

OPTIM_LINKFLAGS=
DEBUG_LINKFLAGS=-g

ifeq ($(BUILD),debug)
CFLAGS = $(DEBUG_CFLAGS)  $(EXTRA_CFLAGS)  -I$(INCLUDE_DIR) 
else # release version
CFLAGS = $(OPTIM_CFLAGS)  $(EXTRA_CFLAGS)  -I$(INCLUDE_DIR) 
endif 



#** LINKFLAGS:  add specific libraries and switches depending on platforms
# possibly this should check on AIX as well
# if so, we should additionally check on ifeq($(CC),gcc)
# as -Xlinker is only appropriate for gcc
ifeq ($(SYSTEM),CC)
LINKFLAGS=-Xlinker -bbigtoc $(EXTRA_LINKFLAGS) $(EXTRA_LIBS)
else
LINKFLAGS=$(EXTRA_LINKFLAGS) $(EXTRA_LIBS) 
endif

ifeq ($(BUILD),debug)
LINKFLAGS+= $(DEBUG_LINKFLAGS)
else # release version
LINKFLAGS+= $(OPTIM_LINKFLAGS) 
endif 


#******** system libraries

SYS_LIBS = -lm
 
# add any others specific to your system using the SYSTEM macros


#********* macros for compiler options
# see below why we need them

ifeq ("$(IS_MS_VC)","")
# the 'normal' case of cc, ar, etc.

# flag to be passed to $(CXX) and $(CC) for specifying the name of the object (.o) file
O_OUTFLAG := -o
# extension for object files
O_SUFFIX := .o
# flag to be passed to $(LINK) for specifying the name of the executable
# warning: there has to be a space after the -o. Otherwise, ld on OSX complains.
# Note that Visual Studio does not work when there is a space after its EXE_OUTFLAG...
EXE_OUTFLAG := -o 
# note: extension for executables is handled somewhat differently
# CYGWIN make (in Unix mode) appends the .exe in targets etc, so we need 
# $(EXE_SUFFIX) only for copying, deleting files etc

# extension for archives
LIB_SUFFIX:=.a
# usually, on Unix, libraries are called libsomething.a, such that one could do -lsomething
LIB_PREFIX:=lib

else

# various settings for the Microsoft Visual C++ compiler
# Unfortunately, this compiler has non-standard options, so we need
# various macros such that the rest of the Makefiles can be general 
# (although a bit less readable unfortunately)

# KT has put other options here as well, to avoid cluttering this file
# with stuff that most people don't want to see.

# Note that we set CC, LINK and AR here appropriately, see you only need to set 
# CXX=cl for all this to jump into action.

CC=$(CXX)

O_OUTFLAG:=/Fo
O_SUFFIX:=.obj
LINK=link 
#EXE_OUTFLAG=/Fe
EXE_OUTFLAG:=/out:
LIB_SUFFIX:=.lib
LIB_PREFIX:=
AR:=link
ARFLAGS=-lib -nologo 
AR_OUTFLAG=-out:

# Normally, you should set the LIB and INCLUDE environment variables. 
# If not, you could specify the location of your Visual Studio files by hand
#MSVCLOC:=c:/Program Files/Microsoft Visual C++ Toolkit 2003/
EXTRA_LINKFLAGS= /nologo  
#/libpath:"${MSVCLOC}lib"

# get rid of -lm above
SYS_LIBS=  

DEBUG_CFLAGS=/Z7 /Od /D _DEBUG /GS /GZ /MLd 
OPTIM_CFLAGS=/D NDEBUG /Ox /ML 
EXTRA_CFLAGS=/nologo /G6 /W3 /GR /GX /D "_WINDOWS"  /D "WIN32" 
# /I"${MSVCLOC}include" 

OPTIM_LINKFLAGS= /incremental:no
DEBUG_LINKFLAGS= /debug

endif # end of settings for MS VC

#********  customisation
# include local configuration file, possibly overriding any of the above
# use -include to not warn when it does not exist

-include $(WORKSPACE)/local/config.mk



#***********************************************************************
# from here on, things are noncustomisable anymore 
# (except by editing this file, which is strongly discouraged for 'normal' users)
#***********************************************************************




#******* LLN matrix libraries
# check if we find the Louvain la Neuve distribution by looking for matrix.h
ifeq ($(wildcard $(LLN_INCLUDE_DIR)/matrix.h),$(LLN_INCLUDE_DIR)/matrix.h)
  ifneq ($(HAVE_LLN_MATRIX),0)

  # yes, the LLN files seem to be there, so we can compile 
  HAVE_LLN_MATRIX=1
  CFLAGS +=  -I $(LLN_INCLUDE_DIR) -D HAVE_LLN_MATRIX
  EXTRA_LIBS += $(LLN_LIB_DIR)/$(LIB_PREFIX)ecat$(LIB_SUFFIX)
  ifeq ($(SYSTEM),SUN)
     SYS_LIBS += -lnsl -lsocket
  endif
  endif
endif


#****** other configuration checks
# make a variable that allows us to test if we have gcc 3.0 or later
# If so, this will allow us to simplify generating the dependencies a lot!
IS_GCC_3:=$(shell $(CXX) -v 2>&1 |grep "gcc version [3456789]")

# make a variable that allows us to test if we have GNU ar
# If so, this we do not run ranlib but use the s option to save some time
IS_GNU_AR:=$(shell $(AR) --version 2>&1 |grep "GNU ar")
ifneq ("$(IS_GNU_AR)","")
  # $(warning Using GNU ar)
  ARFLAGS:=rs
else
  # $(warning Not using GNU ar)
endif

# a variable that will be used to get the real name of an executable
# this is necessary on Windows.
# CYGWIN make automatically appends .exe to executable files in targets etc, but not
# commands such as cp
ifeq ($(SYSTEM),CYGWIN)
EXE_SUFFIX := .exe
else
#EXE_SUFFIX=
endif


