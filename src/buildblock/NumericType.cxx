//
// $Id$
//

/*!
  \file 
 
  \brief implementations for the stir::NumericType class

  \author Kris Thielemans
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
/* 
  History:
  - first version Kris Thielemans

  - KT 15/12/99 added get_Interfile_info
 */
#include "stir/NumericType.h"
#include "stir/NumericInfo.h"

START_NAMESPACE_STIR

NumericType::NumericType(const string& number_format, const size_t size_in_bytes)
{ 
  bool it_is_signed;
  bool it_is_integral;
  
  if (number_format == "signed integer")
  {
    it_is_signed = true;
    it_is_integral = true;
  }
  else if (number_format == "unsigned integer")
  {
    it_is_signed = false;
    it_is_integral = true;
  }  
  else if (number_format == "float")
  {
    it_is_signed = true;
    it_is_integral = false;
  }
  
  else if (number_format == "bit")
  {
    id = BIT;
    return;
  }
  else
  {
    // set to some values which are guaranteed to break later on
    it_is_signed = false;
    it_is_integral = false;
  }
  
  // set up default value
  id = UNKNOWN_TYPE;
  
  // rely on enum<->int conversions
  for (int t=1; t < (int)UNKNOWN_TYPE; t++)
  {
    const NumericType type = (Type)t;
    if (it_is_signed == type.signed_type() &&
        it_is_integral == type.integer_type() &&
        size_in_bytes == type.size_in_bytes())
    { 
      id = (Type)t;
      return;
    }
  }
  
}		


void 
NumericType::get_Interfile_info(string& number_format, 
				size_t& size_in_bytes_v) const
{
  size_in_bytes_v = size_in_bytes();
  
  switch(id)
  {
  case BIT:
    number_format = "bit"; break;
  case  SCHAR:
  case  SHORT:
  case  INT:
  case  LONG:
    number_format = "signed integer"; break;
  case  UCHAR:
  case  USHORT:
  case  UINT:
  case  ULONG:
    number_format = "unsigned integer"; break;
  case  FLOAT:
  case  DOUBLE:
    number_format = "float"; break;
  case UNKNOWN_TYPE:
    number_format = "unknown"; break;
  }
}



size_t NumericType::size_in_bytes() const
    { 
  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
      switch(id)
      {
      case BIT:
	// TODO sensible value ?
	return 0;
#define CASE(NUMERICTYPE)                                \
    case NUMERICTYPE :                                   \
      return NumericInfo<TypeForNumericType<NUMERICTYPE >::type>().size_in_bytes();

      // now list cases that we want
      CASE(NumericType::SCHAR);
      CASE(NumericType::UCHAR);
      CASE(NumericType::SHORT);
      CASE(NumericType::USHORT);
      CASE(NumericType::INT);
      CASE(NumericType::UINT);
      CASE(NumericType::LONG);
      CASE(NumericType::ULONG);
      CASE(NumericType::FLOAT);
      CASE(NumericType::DOUBLE);
#undef CASE
      case UNKNOWN_TYPE:
	return 0;
      }
      // we never get here, but VC++ wants a return nevertheless
      return 0;
    }

size_t NumericType::size_in_bits() const
    { return CHAR_BIT * size_in_bytes(); }

bool NumericType::signed_type() const
    { 
      switch(id)
      {
      case BIT:
	return false;
#define CASE(NUMERICTYPE)                                \
    case NUMERICTYPE :                                   \
      return NumericInfo<TypeForNumericType<NUMERICTYPE >::type>().signed_type();

      // now list cases that we want
      CASE(NumericType::SCHAR);
      CASE(NumericType::UCHAR);
      CASE(NumericType::SHORT);
      CASE(NumericType::USHORT);
      CASE(NumericType::INT);
      CASE(NumericType::UINT);
      CASE(NumericType::LONG);
      CASE(NumericType::ULONG);
      CASE(NumericType::FLOAT);
      CASE(NumericType::DOUBLE);
#undef CASE
      case UNKNOWN_TYPE:
	return false;
      }
      // we never get here, but VC++ wants a return nevertheless
      return false;
    }

bool NumericType::integer_type() const
    { 
      switch(id)
      {
      case BIT:
	return true;
#define CASE(NUMERICTYPE)                                \
    case NUMERICTYPE :                                   \
      return NumericInfo<TypeForNumericType<NUMERICTYPE >::type>().integer_type();

      // now list cases that we want
      CASE(NumericType::SCHAR);
      CASE(NumericType::UCHAR);
      CASE(NumericType::SHORT);
      CASE(NumericType::USHORT);
      CASE(NumericType::INT);
      CASE(NumericType::UINT);
      CASE(NumericType::LONG);
      CASE(NumericType::ULONG);
      CASE(NumericType::FLOAT);
      CASE(NumericType::DOUBLE);
#undef CASE
      case UNKNOWN_TYPE:
	return false;

      }
      // we never get here, but VC++ wants a return nevertheless
      return false;
    }
END_NAMESPACE_STIR
