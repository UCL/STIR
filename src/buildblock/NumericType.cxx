//
// $Id$: $Date$
//

/* 
  Implementation of methods of class NumericType
  
  History:
  - first version Kris Thielemans

  - KT 15/12/99 added get_Interfile_info
   */
#include "NumericType.h"
#include "NumericInfo.h"

NumericType::NumericType(const string number_format, const size_t size_in_bytes)
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
      case  SCHAR:
	return NumericInfo<signed char>().size_in_bytes();
      case  UCHAR:
	return NumericInfo<unsigned char>().size_in_bytes();
      case  SHORT:
	return NumericInfo<short>().size_in_bytes();
      case  USHORT:
	return NumericInfo<unsigned short>().size_in_bytes();
      case  INT:
	return NumericInfo<int>().size_in_bytes();
      case  UINT:
	return NumericInfo<unsigned int>().size_in_bytes();
      case  LONG:
	return NumericInfo<long>().size_in_bytes();
      case  ULONG:
	return NumericInfo<unsigned long>().size_in_bytes();
      case  FLOAT:
	return NumericInfo<float>().size_in_bytes();
      case  DOUBLE:
	return NumericInfo<double>().size_in_bytes();
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
  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
      switch(id)
      {
      case BIT:
	return false;
      case  SCHAR:
	return NumericInfo<signed char>().signed_type();
      case  UCHAR:
	return NumericInfo<unsigned char>().signed_type();
      case  SHORT:
	return NumericInfo<short>().signed_type();
      case  USHORT:
	return NumericInfo<unsigned short>().signed_type();
      case  INT:
	return NumericInfo<int>().signed_type();
      case  UINT:
	return NumericInfo<unsigned int>().signed_type();
      case  LONG:
	return NumericInfo<long>().signed_type();
      case  ULONG:
	return NumericInfo<unsigned long>().signed_type();
      case  FLOAT:
	return NumericInfo<float>().signed_type();
      case  DOUBLE:
	return NumericInfo<double>().signed_type();
      case UNKNOWN_TYPE:
	return false;
      }
      // we never get here, but VC++ wants a return nevertheless
      return false;
    }

bool NumericType::integer_type() const
    { 
  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
      switch(id)
      {
      case BIT:
	return true;
      case  SCHAR:
	return NumericInfo<signed char>().integer_type();
      case  UCHAR:
	return NumericInfo<unsigned char>().integer_type();
      case  SHORT:
	return NumericInfo<short>().integer_type();
      case  USHORT:
	return NumericInfo<unsigned short>().integer_type();
      case  INT:
	return NumericInfo<int>().integer_type();
      case  UINT:
	return NumericInfo<unsigned int>().integer_type();
      case  LONG:
	return NumericInfo<long>().integer_type();
      case  ULONG:
	return NumericInfo<unsigned long>().integer_type();
      case  FLOAT:
	return NumericInfo<float>().integer_type();
      case  DOUBLE:
	return NumericInfo<double>().integer_type();
      case UNKNOWN_TYPE:
	return false;

      }
      // we never get here, but VC++ wants a return nevertheless
      return false;
    }
