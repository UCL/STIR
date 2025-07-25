/*
    Copyright (C) 2016-2017, 2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_config__H__
#define __stir_config__H__
/*!
  \file 
  \ingroup buildblock 
  \brief basic configuration include file 

  This file will be used by CMake to create stir/config.h based on local settings and CMake options.

  \author Kris Thielemans
*/

namespace stir {
  /*!
    \name Preprocessor symbols with version information
    \ingroup buildblock
    Values are set by CMake from the main CMakeLists.txt.
  */
  //@{
  #define STIR_VERSION 060200
  #define STIR_VERSION_STRING "6.2.0"
  /*! \def STIR_VERSION
      \brief numerical amalgation of the version info, as in 030100 (for version 3.1.0)
  */
  /*! \def STIR_VERSION_STRING
      \brief a string with the version info, as in "3.1.0" (for version 3.1.0)
  */
  
  //@}
}

// preprocessor symbols with location of installed files
// do NOT use directly, but use functions in find_STIR_config.h instead
#define STIR_CONFIG_DIR "/usr/local/share/STIR-6.2/config"

#define STIR_DOC_DIR "/usr/local/share/doc/STIR-6.2"

#if defined(_MSC_VER)
#include "stir/config/visualc.h"
#endif
#if defined(__GNUC__)
#include "stir/config/gcc.h"
#endif

#include "boost/config.hpp"

// 2 variables set via CMake
/* #undef BIG_ENDIAN_BYTE_ORDER_FROM_CMAKE */
#define LITTLE_ENDIAN_BYTE_ORDER_FROM_CMAKE

/* #undef HAVE_ECAT */
#ifdef HAVE_ECAT
#define HAVE_LLN_MATRIX
#endif

/* #undef HAVE_CERN_ROOT */

/* #undef HAVE_UPENN */

#define HAVE_HDF5

#define HAVE_ITK

#define HAVE_JSON

/* #undef STIR_WITH_NiftyPET_PROJECTOR */

#define STIR_WITH_CUDA

#define STIR_WITH_Parallelproj_PROJECTOR
#define parallelproj_built_with_CUDA

#define STIR_OPENMP

/* #undef STIR_MPI */

#define nlohmann_json_FOUND "1"

/* #undef STIR_USE_BOOST_SHARED_PTR */

/* #undef STIR_NO_UNIQUE_PTR */

#define HAVE_SYSTEM_GETOPT

/* #undef STIR_DEFAULT_PROJECTOR_AS_V2 */
#ifndef STIR_DEFAULT_PROJECTOR_AS_V2
#define USE_PMRT
#endif

/* #undef STIR_PROJECTORS_AS_V3 */
/* #undef STIR_ROOT_ROTATION_AS_V4 */
/* #undef STIR_LEGACY_IGNORE_VIEW_OFFSET */

#define STIR_TOF 1
/*! \def STIR_TOF
  \brief Defined if TOF capabilities are enabled.
*/

/* #undef MINI_STIR */

#endif //  __stir_config__H__
