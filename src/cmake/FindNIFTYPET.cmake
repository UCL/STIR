# Copyright 2019 University College London

# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

# Find the base folder containing prjf.h
find_path(NIFTYPET_INCLUDE_DIR "niftypet/nipet/prj/src/prjf.h")

find_library(NIFTYPET_PETPRJ_LIB "petprj"
  DOC "NIFTYPET projector library")
find_library(NIFTYPET_MMR_AUXE_LIB "mmr_auxe"
  DOC "NIFTYPET aux library")
find_library(NIFTYPET_MMR_LMPROC_LIB "mmr_lmproc"
  DOC "NIFTYPET lmproc library")

# handle the QUIETLY and REQUIRED arguments and set NIFTYPET_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NIFTYPET 
  "NIFTYPET components not found. If you do have it, set the missing variables"
  NIFTYPET_INCLUDE_DIR NIFTYPET_PETPRJ_LIB NIFTYPET_MMR_AUXE_LIB NIFTYPET_MMR_LMPROC_LIB)

SET(NIFTYPET_LIBRARIES 
  ${NIFTYPET_PETPRJ_LIB}
  ${NIFTYPET_MMR_AUXE_LIB}
  ${NIFTYPET_MMR_LMPROC_LIB}
)

SET(NIFTYPET_INCLUDE_DIRS 
  "${NIFTYPET_INCLUDE_DIR}/niftypet/nipet"
  "${NIFTYPET_INCLUDE_DIR}/niftypet/nipet/dinf"
  "${NIFTYPET_INCLUDE_DIR}/niftypet/nipet/lm/src"
  "${NIFTYPET_INCLUDE_DIR}/niftypet/nipet/prj/src"
  "${NIFTYPET_INCLUDE_DIR}/niftypet/nipet/sct/src"
  "${NIFTYPET_INCLUDE_DIR}/niftypet/nipet/src"
)

MARK_AS_ADVANCED(NIFTYPET_INCLUDE_DIR NIFTYPET_PETPRJ_LIB NIFTYPET_MMR_AUXE_LIB NIFTYPET_MMR_LMPROC_LIB)