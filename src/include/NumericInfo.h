#ifndef __NUMERICINFO_H_
#define __NUMERICINFO_H_
//
// $Id$: $Date$
//

/*
  class NumericInfo<NUMBER> defines properties for the type NUMBER.
     Lot's of expansion is possible here.
     The idea comes from the Modena C++ (and VC++) library where they have a 
     similar class numeric_limits. I have no idea if this is a part of the new
     ANSI C++ standard, but g++ doesn't have it yet.. However, if so,
     I should probably rename a few things. (TODO)

  History : 
  first version by Kris Thielemans

  KT 15/12/99
  reordered member function declarations to allow easier inlining
  moved NumericType and ByteOrder to separate files
 */ 

#include "NumericType.h"

// this code uses' #define's  as specified by the ANSI C standard
#include <limits.h>
#include <float.h>
#ifndef __MSL__
#include <sys/types.h>
#else
#include <size_t.h>
#endif



// the general framework of NumericInfo
// This declaration is essentially empty, but needed to use template syntax.
// It could be better to have some error in here, flagging (at compile time)
// when you use it for a type which is not yet in our list. With the code 
// below, you'll see a linker error only, something like
// 'undefined reference NumericInfo<complex>::signed_type()'
template <class NUMBER>
class NumericInfo
{
public:
  inline bool signed_type() const;
  inline bool integer_type() const;
  // as reported by sizeof(), so really size_in_sizeof_char
  inline size_t size_in_bytes() const;
  inline size_t size_in_bits() const;

  inline NUMBER max_value() const;
  inline NUMBER min_value() const;

  inline NumericType type_id() const;
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
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
  size_t size_in_bytes() const
   { return sizeof(type); }
  size_t size_in_bits() 
    { return CHAR_BIT*size_in_bytes(); }
  type max_value() const
    { return DBL_MAX; }
  type min_value() const
    { return -DBL_MAX; }
  NumericType type_id() const
    { return NumericType::DOUBLE; }
};



#endif
