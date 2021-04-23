//
//
/*!
  \file
  \ingroup buildblock
  
  \brief Import of std::shared_ptr, std::dynamic_pointer_cast and
  std::static_pointer_cast (or corresponding boost versions if 
  STIR_USE_BOOST_SHARED_PTR is set, i.e. normally when std::shared_ptr doesn't exist) 
  into the stir namespace.        
*/         
/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SHARED_PTR__
#define __stir_SHARED_PTR__

#include "stir/common.h"
#if defined(STIR_USE_BOOST_SHARED_PTR)
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
