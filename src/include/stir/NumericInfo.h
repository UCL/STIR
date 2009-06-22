#ifndef __NUMERICINFO_H_
#define __NUMERICINFO_H_
//
//  $Id$
//
/*! 
  \file
  \ingroup buildblock 
  \brief  This file declares the class stir::NumericInfo.

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
// TODO some of these members could be made static

/*
  History : 
  first version by Kris Thielemans

  KT 15/12/99
  reordered member function declarations to allow easier inlining
  moved NumericType and ByteOrder to separate files

  KT 18/02/2000 use new syntax for template specialisations: template<> {class blabla};
 */ 

#include "stir/NumericType.h"

// this code uses #define's  as specified by the ANSI C standard
#include <climits>
#include <cfloat>
#ifndef __MSL__
#include <sys/types.h>
#else
#include <size_t.h>
#endif

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief class NumericInfo<NUMBER> defines properties for the type NUMBER.

     
  The idea comes the ANSI C++ class \c std::numeric_limits, but 
  some compiler didn't have this yet when PARAPET/STIR was started.
  This current class should disappear at some point.

  Note that this general template is essentially empty, but it is needed to use
  template syntax.
 \warning It could be better to have some error in here, flagging (at compile
  time) when you use it for a type which is not yet in our list. With the 
  current implementation, you will see a linker error only, something like
 'undefined reference NumericInfo<complex>::signed_type()'
*/
template <class NUMBER>
class NumericInfo
{
public:
  inline bool signed_type() const;

  inline bool integer_type() const;
  //! Size in 'bytes' as reported by sizeof(), so really size_in_sizeof_char
  inline size_t size_in_bytes() const;
  inline size_t size_in_bits() const;

  inline NUMBER max_value() const;
  inline NUMBER min_value() const;

  inline NumericType type_id() const;
};



// Below are the actual details filled in.

//! Basic properties of signed char
template<>
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

//! Basic properties of unsigned char
template<>
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

//! Basic properties of signed short
template<>
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

//! Basic properties of unsigned short
template<>
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


//! Basic properties of signed int
template<>
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

//! Basic properties of unsigned int
template<>
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


//! Basic properties of signed long
template<>
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

//! Basic properties of unsigned long
template<>
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

//! Basic properties of float
template<>
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

//! Basic properties of double
template<>
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

END_NAMESPACE_STIR

#endif
