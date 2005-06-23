// $Id$
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup Array 
  \brief non-inline implementations for the Array class 

  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$

  This file could be empty. However, it contains
  contains instantiations for some common cases. 
  This might reduce the size of the executable a bit
  if the compiler cannot inline certain functions.
*/

#include "stir/Array.h"

START_NAMESPACE_STIR


/**************************************************
 instantiations
 **************************************************/

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
// add any other types you need
template class Array<1,signed char>;
template class Array<1,short>;
template class Array<1,unsigned short>;
template class Array<1,float>;
#endif

template class Array<2,signed char>;
template class Array<2,short>;
template class Array<2,unsigned short>;
template class Array<2,float>;

template class Array<3, signed char>;
template class Array<3, short>;
template class Array<3,unsigned short>;
template class Array<3,float>;

template class Array<4, short>;
template class Array<4,unsigned short>;
template class Array<4,float>;

END_NAMESPACE_STIR
