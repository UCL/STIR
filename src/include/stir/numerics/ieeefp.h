//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_numerics_ieeefp_H__
#define __stir_numerics_ieeefp_H__
/*!
  \file
  \ingroup numerics
  
  \brief Definition of work-around macros STIR_isnan and STIR_finite for a few non-portable IEEE floating point functions     

  \warning The setting of these macros is tricky and only tested on a few systems.
  Hopefully, somebody else can find a better way. For example,
  http://std.dkuug.dk/jtc1/sc22/wg21/docs/papers/2004/n1568.htm

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#include "stir/common.h"


#if defined(_MSC_VER)

#  include <float.h>
#  define STIR_isnan _isnan 
#  define STIR_finite _finite 

#else

#  include <cmath>

  // attempt to get to find isnan but it might not work on some systems
  // terrible hack to try and find isnan in std
  // will only work for gcc
#  if _GLIBCPP_USE_C99 && !_GLIBCPP_USE_C99_FP_MACROS_DYNAMIC
#    define STIR_isnan std::isnan
#    define STIR_finite std::isfinite
#  else
#    if defined(HAVE_IEEEFP_H) // this macro might be set (but probably only by configure)
#      include <ieeefp.h>
#      define STIR_isnan isnan
#      define STIR_finite finite
#    endif
#  endif

#endif

#if !defined(STIR_isnan) 
# if defined(isnan)
  #define STIR_isnan isnan
# else //portable version
  // according to IEEE rules if x is NaN, then x!=x
  // so, the following will work even on non-IEEE systems
  #define STIR_isnan(x) (x)!=(x)
# endif
#endif

#if !defined(STIR_finite) 
# if defined(finite)
#  define STIR_finite finite
# else //portable version
  // we give up and say all numbers are finite
  #define STIR_finite(x) true
# endif
#endif

#ifdef DOXYGEN_SKIP // only when running doxygen
  // doxygen doesn't execute above preprocessor commands, but then doesn't generate documentation
  // so define something here
  #define STIR_finite finite
  #define STIR_isnan isnan
#endif
/*!
\def STIR_isnan(x)
A (hopefully) portable way to call isnan. Current implementation does not always
find isnan and then reverts to (x)!=(x), which according to IEEE math should work as well.
*/

/*!
\def STIR_finite(x)
A (hopefully) portable way to call \c finite. But problems can occur on your system.
Current implementation does not always find \c finite and then reverts to \c true.
*/


#endif
