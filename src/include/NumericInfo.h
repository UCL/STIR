#ifndef NUMERICINFO_H
#define NUMERICINFO_H
/*
  $Id$ : $Date$

  This file defines two classes : 
  - NumericType
     a class with names for some numeric types
     a first start of propries is there as well, TODO expand to others
  - NumericInfo<NUMBER>
     a class that defines properties for the type NUMBER.
     Lot's of expansion is possible here.
     The idea comes from the Modena C++ library where they have a 
     similar class numeric_limits. I have no idea if this is a part of the new
     ANSI C++ standard, but g++ doesn't have it yet.. However, if so,
     I should probably rename a few things. (TODO)

  History : first version by KT
 */ 

// for PETerror
#include "pet_common.h"

// KT this code uses' #define's  as specified by the ANSI C standard
#include <limits.h>
#include <float.h>
#ifndef __MSL__
#include <sys/types.h>
#else
#include <size_t.h>
#endif

class NumericType
{
public:
  enum Type { BIT, SCHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, 
	 FLOAT, DOUBLE, UNKNOWN } id;
  NumericType(Type t)
    : id(t) 
    {}

  bool operator==(NumericType type) const
    { return id == type.id; }
  // as reported by sizeof(), so really size_in_sizeof_char
  // implementation is at the end of the file, as it uses NumericInfo below
  inline size_t size_in_bytes() const;
  size_t size_in_bits() const
    { return CHAR_BIT * size_in_bytes(); }

};

// the general framework
// This declaration is essentially empty, but needed to use template syntax.
// It could be better to have some error in here, flagging (at compile time)
// when using it for a type which is not yet in our list. With the code 
// below, you'll see a linker error only, something like
// 'undefined reference NumericInfo<complex>::signed_type()'
//TODO all functions should be const
template <class NUMBER>
class NumericInfo
{
public:
  bool signed_type() const;
  bool integer_type() const;
  size_t size_in_bits() const;
  // as reported by sizeof(), so really size_in_sizeof_char
  size_t size_in_bytes() const;

  NUMBER max_value() const;
  NUMBER min_value() const;

  NumericType type_id() const;
};



// Below are the actual details filled in.
class NumericInfo<signed char>
{
  typedef signed char type;
public:
  bool signed_type() const
    { return true; }
  bool integer_type() const
    { return true; }
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  type max_value() const
    { return SCHAR_MAX; }
  type min_value() const
    { return SCHAR_MIN; }
  NumericType type_id() const
    { return NumericType::SCHAR; }
};

class NumericInfo<unsigned char>
{
  typedef unsigned char type;
public:
  bool signed_type() const
    { return false; }
  bool integer_type() const
    { return true; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return UCHAR_MAX; }
  type min_value() const
    { return 0; }
  NumericType type_id() const
    { return NumericType::UCHAR; }
};

class NumericInfo<signed short>
{
  typedef signed short type;
public:
  bool signed_type() const
    { return true; }
  bool integer_type() const
    { return true; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return SHRT_MAX; }
  type min_value() const
    { return SHRT_MIN; }
  NumericType type_id() const
    { return NumericType::SHORT; }
};


class NumericInfo<unsigned short>
{
  typedef unsigned short type;
public:
  bool signed_type() const
    { return false; }
  bool integer_type() const
    { return true; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return USHRT_MAX; }
  type min_value() const
    { return 0; }
  NumericType type_id() const
    { return NumericType::USHORT; }
};


class NumericInfo<signed int>
{
  typedef signed int type;
public:
  bool signed_type() const
    { return true; }
  bool integer_type() const
    { return true; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return INT_MAX; }
  type min_value() const
    { return INT_MIN; }
  NumericType type_id() const
    { return NumericType::INT; }
};

class NumericInfo<unsigned int>
{
  typedef unsigned int type;
public:
  bool signed_type() const
    { return false; }
  bool integer_type() const
    { return true; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return UINT_MAX; }
  type min_value() const
    { return 0; }
  NumericType type_id() const
    { return NumericType::UINT; }
};


class NumericInfo<signed long>
{
  typedef signed long type;
public:
  bool signed_type() const
    { return true; }
  bool integer_type() const
    { return true; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return LONG_MAX; }
  type min_value() const
    { return LONG_MIN; }
  NumericType type_id() const
    { return NumericType::LONG; }
};

class NumericInfo<unsigned long>
{
  typedef unsigned long type;
public:
  bool signed_type() const
    { return false; }
  bool integer_type() const
    { return true; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return ULONG_MAX; }
  type min_value() const
    { return 0; }
  NumericType type_id() const
    { return NumericType::ULONG; }

};

class NumericInfo<float>
{
  typedef float type;
public:
  bool signed_type() const
    { return true; }
  bool integer_type() const
    { return false; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return FLT_MAX; }
  type min_value() const
    { return -FLT_MAX; }
  NumericType type_id() const
    { return NumericType::FLOAT; }
};

class NumericInfo<double>
{
  typedef double type;
public:
  bool signed_type() const
    { return true; }
  bool integer_type() const
    { return false; }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  size_t size_in_bytes() const
   { return sizeof(type); }
  type max_value() const
    { return DBL_MAX; }
  type min_value() const
    { return -DBL_MAX; }
  NumericType type_id() const
    { return NumericType::DOUBLE; }
};


//***********************************************************************

inline size_t NumericType::size_in_bytes() const
    { 
  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
      switch(id)
      {
      case  SCHAR:
	return NumericInfo<signed char>().size_in_bytes();
      case  UCHAR:
	return NumericInfo<unsigned char>().size_in_bytes();
      case  SHORT:
	return NumericInfo<short>().size_in_bytes();
      case  USHORT:
	return NumericInfo<unsigned short>().size_in_bytes();
      case  FLOAT:
	return NumericInfo<float>().size_in_bytes();
      default:
	// TODO
	PETerror("type not yet supported"); Abort();
      }
    }

#endif
