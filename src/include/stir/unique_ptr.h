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

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_UNIQUE_PTR__
#define __stir_UNIQUE_PTR__

// include this as stir/common.h has to be included by any stir .h file
#include "stir/common.h"
#include <memory>
// simply use std::unique_ptr
namespace stir {
  using std::unique_ptr;
}

#endif
