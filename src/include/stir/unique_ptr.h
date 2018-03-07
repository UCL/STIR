/*!
  \file
  \ingroup buildblock
  
  \brief Import of std::unique_ptr into the stir namespace, together with 
  work-arounds for other compilers.

  If std::unique doesn't exist, we will define unique_ptr to auto_ptr. This is
  dangerous of course. If you have problems with that, tell your compiler to use at least C++11
  (normally by adding the -std=c++11 flag)

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

// KT attempted to use boost::movelib::unique_ptr, which is claimed to be compatible with
// std::unique_ptr but even for Cxx03 compilers. However, end 2016 this still generated errors on OSX Sierra
// with CLang (it could not return a unique_ptr).
// So, this is attempt is now disabled.
#if 0
#include <boost/version.hpp>
#if (BOOST_VERSION >= 105700)
// Boost is recent enough to have a drop-in replacement
#include <boost/move/unique_ptr.hpp>
namespace stir {
  using boost::movelib::unique_ptr;
}
#endif
#endif

// desperate measures. We will use a #define to auto_ptr.
// Caveat:
// This trick is likely to break on OSX as it will generate conflicts between
// this define and Apple's non-C++-11 compliant unique_ptr.
// (Reason: BOOST_NO_CXX11_SMART_PTR is set, so we define unique_ptr here, but some include files will still
// have unique_ptr with 2 template arguments, while auto_ptr needs only one).
// Best solution: tell your compiler to use C++-11 or later (normally this means adding -std=c++11)

// We first include a bunch of system files which use std::unique_ptr such that we don't have a conflict.
// You might have to add a few more...
#include <map>
#include <vector>
#include <string>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <deque>
#include <boost/functional/hash/extensions.hpp>
#include <boost/get_pointer.hpp>
#include <boost/smart_ptr/detail/shared_count.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

#define unique_ptr auto_ptr
using std::auto_ptr;
#endif

#endif
