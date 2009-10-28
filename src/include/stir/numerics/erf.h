//
// $Id$
//
/*
    Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup numerics
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
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

