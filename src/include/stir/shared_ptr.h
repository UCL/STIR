//
//
/*!
  \file
  \ingroup buildblock
  
  \brief Import of boost::shared_ptr into stir namespace
          
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
#include "boost/shared_ptr.hpp"
namespace stir {
using boost::shared_ptr;
}

#endif
