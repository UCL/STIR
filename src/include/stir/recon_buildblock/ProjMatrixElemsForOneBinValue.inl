//
// $Id$
//
/*!
  \file
  \ingroup projection
  
  \brief Inline implementations for class stir::ProjMatrixElemsForOneBinValue
    
  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project
      
  $Date$        
  $Revision$
*/
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

#include "stir/Coordinate3D.h"

//for SHRT_MAX etc
#ifndef NDEBUG
#include <climits>
#endif

START_NAMESPACE_STIR

ProjMatrixElemsForOneBinValue::
ProjMatrixElemsForOneBinValue(const BasicCoordinate<3,int>& coords,
                                const float ivalue)
    : c3(static_cast<short>(coords[3])),
      c2(static_cast<short>(coords[2])),
      c1(static_cast<short>(coords[1])),
      value(ivalue)
{
  assert(coords[3] <= SHRT_MAX);
  assert(coords[3] >= SHRT_MIN);
  assert(coords[2] <= SHRT_MAX);
  assert(coords[2] >= SHRT_MIN);
  assert(coords[1] <= SHRT_MAX);
  assert(coords[1] >= SHRT_MIN);
}  

ProjMatrixElemsForOneBinValue::
ProjMatrixElemsForOneBinValue()
    : c3(0),
      c2(0),
      c1(0),
      value(0)
{}  

BasicCoordinate<3,int> 
ProjMatrixElemsForOneBinValue::
get_coords() const
{
  return Coordinate3D<int>(c1,c2,c3);
}

int 
ProjMatrixElemsForOneBinValue::
coord1() const
{ return static_cast<int>(c1); }

int 
ProjMatrixElemsForOneBinValue::
coord2() const
{ return static_cast<int>(c2); }

int 
ProjMatrixElemsForOneBinValue::
coord3() const
{ return static_cast<int>(c3); }

float 
ProjMatrixElemsForOneBinValue::
get_value() const
{ return value; }

  
ProjMatrixElemsForOneBinValue& 
ProjMatrixElemsForOneBinValue::
operator+=(const ProjMatrixElemsForOneBinValue& el2)
{
  assert(get_coords() == el2.get_coords());
  value += el2.value;
  return *this;
}

ProjMatrixElemsForOneBinValue& 
ProjMatrixElemsForOneBinValue::
operator+=(const float d)
{
  value += d;
  return *this;
}

ProjMatrixElemsForOneBinValue& 
ProjMatrixElemsForOneBinValue::
operator*=(const float d)
{
  value *= d;
  return *this;
}

ProjMatrixElemsForOneBinValue& 
ProjMatrixElemsForOneBinValue::
operator/=(const float d)
{
  value /= d;
  return *this;
}

bool 
ProjMatrixElemsForOneBinValue::
coordinates_equal(const ProjMatrixElemsForOneBinValue& el1, const ProjMatrixElemsForOneBinValue& el2)
{
  return el1.c3==el2.c3 && el1.c2==el2.c2 && el1.c1==el2.c1;
}

bool 
ProjMatrixElemsForOneBinValue::
coordinates_less(const ProjMatrixElemsForOneBinValue& el1, const ProjMatrixElemsForOneBinValue& el2)
{
  return el1.c1<el2.c1 ||
      (el1.c1==el2.c1&& (el1.c2<el2.c2 || (el1.c2==el2.c2 && el1.c3<el2.c3)));
}



bool 
operator==(const ProjMatrixElemsForOneBinValue& el1, 
           const ProjMatrixElemsForOneBinValue& el2)
{
  return el1.c3==el2.c3 && el1.c2==el2.c2 && el1.c1==el2.c1 && el1.value==el2.value;
}


bool 
operator<(const ProjMatrixElemsForOneBinValue& el1, 
          const ProjMatrixElemsForOneBinValue& el2) 
{
  return 
    el1.c1<el2.c1 ||
      (el1.c1==el2.c1&& 
         (el1.c2<el2.c2 || 
            (el1.c2==el2.c2 && 
               (el1.c3<el2.c3 || (el1.c3==el2.c3 && el1.value<el2.value)))));
}

END_NAMESPACE_STIR
