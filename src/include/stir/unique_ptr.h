/*!
  \file
  \ingroup buildblock
  
  \brief Import of std::unique_ptr into the stir namespace, together with 
  work-arounds for other compilers.

  If std::unique doesn't exist, attempt to use boost::movelib::unique_ptr. However, that needs
  boost 1.59, so if you don't have that, we will define unique_ptr to auto_ptr. This is
  dangerous of course. If you have problems with that, upgrade your boost version.

  \author Kris Thielemans
*/         
/*
    Copyright (C) 2016, University College London
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

#ifndef __stir_UNIQUE_PTR__
#define __stir_UNIQUE_PTR__

// include this as stir/common.h has to be included by any stir .h file
#include "stir/common.h"
#include <memory>
#if !defined(STIR_NO_UNIQUE_PTR)
// simply use std::unique_ptr
namespace stir {
  using std::unique_ptr;
}
#else
// we need to replace it with something else

#if (BOOST_VERSION >= 105700)
// Boost is recent enough to have a drop-in replacement
#include "boost/move/unique_ptr.hpp"
namespace stir {
  using boost::movelib::unique_ptr;
}
#else
// desperate measures. We will use a #define to auto_ptr.
// Caveat:
// This trick is likely to break on (older?) OSX as it will generate conflicts between
// the define and Apple's non-C++-11 compliant unique_ptr.
// (Reason: BOOST_NO_CXX11_SMART_PTR is set, so we define unique_ptr here, but some include files will still
// have unique_ptr with 2 template arguments, while auto_ptr needs only one).
// Solution: tell your compiler to use C++-11 or later, upgrade Boost to 1.57 or newer, or upgrade your compiler...
#define unique_ptr auto_ptr
using std::auto_ptr;
#endif
#endif

#endif
