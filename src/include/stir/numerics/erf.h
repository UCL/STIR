//
//
/*
    Copyright (C) 2004 - 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup numerics
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/

#include "stir/common.h"

START_NAMESPACE_STIR

/*! \ingroup numerics
   \name A collection of error functions. 
   The erf() is a high precision implementation of the error function.
   The erfc() is the complementary of the erf(), which should be equal to 1-erf(), but with 
   higher precision when erf is close to 1.

   \todo replace with boost::erf
*/
//@{
inline 
double erf(double);
inline
double erfc(double);
//@}

END_NAMESPACE_STIR

#include "stir/numerics/erf.inl"

