//
//
/*!
  \file
  \ingroup buildblock
  
  \brief Import of std::shared_ptr, std::dynamic_pointer_cast and
  std::static_pointer_cast (or corresponding boost versions if 
  std::shared_ptr doesn't exist) into the stir namespace
          
*/         
/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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

#ifndef __stir_SHARED_PTR__
#define __stir_SHARED_PTR__

// include this as stir/common.h has to be included by any stir .h file
#include "stir/common.h"
#if defined(BOOST_NO_CXX11_SMART_PTR)
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include "boost/pointer_cast.hpp"
namespace stir {
  using boost::shared_ptr;
  using boost::dynamic_pointer_cast;
  using boost::static_pointer_cast;
  //! work-around for using std::make_shared on old compilers
#define MAKE_SHARED boost::make_shared
}
#else
#include <memory>
namespace stir {
  using std::shared_ptr;
  using std::dynamic_pointer_cast;
  using std::static_pointer_cast;
  //! work-around for using std::make_shared on old compilers
#define MAKE_SHARED std::make_shared
}
#endif

#endif
