#TODO local stuff

##******* GE IO
## check if we find it by looking for niff.h
#ifneq ($(HAVE_GE_IO),0)
#  ifneq ("$(wildcard $(INCLUDE_DIR)/local/stir/IO/GE/niff.h)","")
#     HAVE_GE_IO=1
#     CFLAGS+=-DSTIR_USE_GE_IO
#  endif
#endif
#
##******* GE RDF library
## check if we find it by looking for GErdfUtils.h
#ifneq ($(HAVE_RDF),0)
#  ifneq ("$(wildcard $(RDF_INCLUDE_DIR)/GErdfUtils.h)","")
#     RDF_LIB=${RDF_LIB_DIR}/$(LIB_PREFIX)GEio$(LIB_SUFFIX)
#     ifeq ("$(wildcard $(RDF_LIB))","")
#        $(error --- Found RDF include files, but no library at $(RDF_LIB). Please build this first, or set HAVE_RDF=0 )
#     endif
#     # $(warning found RDF library)
#     HAVE_RDF=1
#     CFLAGS+=-I ${RDF_INCLUDE_DIR} -DHAVE_RDF
#     # note: this won't work for MS VC
#     EXTRA_LIBS += $(RDF_LIB)
#  endif
#endif
#
##******* GE IE library
## check if we find it by looking for IEUtils.h
#ifneq ($(HAVE_IE),0)
#  ifneq ("$(wildcard $(IE_INCLUDE_DIR)/IEUtils.h)","")
#     IE_LIB=${IE_LIB_DIR}/$(LIB_PREFIX)IEIO$(LIB_SUFFIX)
#     ifeq ("$(wildcard $(IE_LIB))","")
#        $(error --- Found IE include files, but no library at $(IE_LIB). Please build this first, or set HAVE_IE=0 )
#     endif
#     # $(warning found IE library)
#     HAVE_IE=1
#     CFLAGS+=-I ${IE_INCLUDE_DIR} -DHAVE_IE
#     # note: this won't work for MS VC
#     EXTRA_LIBS += $(IE_LIB)
#  endif
#endif
#
#TODO
#EXTRA_CFLAGS+=-D STIR_DEVEL
#STIR_DEVEL=1
#STIR_DEVEL_MOTION=1
#EXTRA_CFLAGS+=-DSCATTER_MOTION 
#
#
message(STATUS "Local TODO")
