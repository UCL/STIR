#ifndef NUMERICINFO_H
#define NUMERICINFO_H
/*
  $Id$ : $Date$

  This file defines two classes : 
  - NumericType
     a class with names for some numeric types
  - NumericInfo<NUMBER>
     a class that defines properties for the type NUMBER.
     Lot's of expansion is possible here.
     The idea comes from the Modena C++ library where they have a 
     similar class NumericLimits. This is possibly a part of the new
     ANSI C++ standard, but g++ doesn't have it yet.. However, if so,
     I should probably rename a few things. (TODO)

  History : first version by KT
 */ 


// KT this code work for ANSI C++ compilers (including gcc and AIX cc)
#include <limits.h>
#include <float.h>

// The enum is inside a class to make the avoid namespace clutter.
class NumericType
{
public:

  enum { BIT, SCHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, 
	 FLOAT, DOUBLE } id;
};


// the general framework
// This declaration is essentially empty, but needed to use template syntax.
// It could be better to have some error in here, flagging (at compile time)
// when using it for a type which is not yet in our list. With the code 
// below, you'll see a linker error only, something like
// 'undefined reference NumericInfo<complex>::signed_type()'
template <class NUMBER>
class NumericInfo
{
public:
  bool signed_type();
  bool integer_type();
  int size_in_bits();
  NUMBER max_value();
  NUMBER min_value();
};

class NumericInfo<signed char>
{
  typedef signed char type;
public:
  bool signed_type()
    { return true; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return SCHAR_MAX; }
  type min_value()
    { return SCHAR_MIN; }
};

class NumericInfo<unsigned char>
{
  typedef unsigned char type;
public:
  bool signed_type()
    { return false; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return UCHAR_MAX; }
  type min_value()
    { return 0; }
};

class NumericInfo<signed short>
{
  typedef signed short type;
public:
  bool signed_type()
    { return true; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return SHRT_MAX; }
  type min_value()
    { return SHRT_MIN; }
};


class NumericInfo<unsigned short>
{
  typedef unsigned short type;
public:
  bool signed_type()
    { return false; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return USHRT_MAX; }
  type min_value()
    { return 0; }
};


class NumericInfo<signed int>
{
  typedef signed int type;
public:
  bool signed_type()
    { return true; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return INT_MAX; }
  type min_value()
    { return INT_MIN; }
};

class NumericInfo<unsigned int>
{
  typedef unsigned int type;
public:
  bool signed_type()
    { return false; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return UINT_MAX; }
  type min_value()
    { return 0; }
};


class NumericInfo<signed long>
{
  typedef signed long type;
public:
  bool signed_type()
    { return true; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return LONG_MAX; }
  type min_value()
    { return LONG_MIN; }
};

class NumericInfo<unsigned long>
{
  typedef unsigned long type;
public:
  bool signed_type()
    { return false; }
  bool integer_type()
    { return true; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return ULONG_MAX; }
  type min_value()
    { return 0; }
};

class NumericInfo<float>
{
  typedef float type;
public:
  bool signed_type()
    { return true; }
  bool integer_type()
    { return false; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return FLT_MAX; }
  type min_value()
    { return -FLT_MAX; }
};

class NumericInfo<double>
{
  typedef double type;
public:
  bool signed_type()
    { return true; }
  bool integer_type()
    { return false; }
  int size_in_bits() 
    { return CHAR_BIT*sizeof(type); }
  type max_value()
    { return DBL_MAX; }
  type min_value()
    { return -DBL_MAX; }
};

#endif
