//
//  $Id$
//
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

#ifndef __stir_NumericType_H__
#define __stir_NumericType_H__

/*!
  \file 
  \ingroup buildblock 
  \brief This file declares the stir::NumericType class.

  \author Kris Thielemans 
  \author PARAPET project

  $Date$
  $Revision$
  */
/*
  Modification History:

  - first version by Kris Thielemans

  - KT 28/07/98 
  added two methods to NumericType:
  inline bool signed_type() const;
  inline bool integer_type() const;
  
  - KT 15/12/99
  removed inlining. Gave troubles with gcc on NT.
  added method that returns Interfile info

  - KT 15/07/04
  added TypeForNumericType
*/

#include "stir/common.h"
#include <string>

START_NAMESPACE_STIR

#ifndef STIR_NO_NAMESPACES
using std::size_t;
using std::string;
#endif

/*!
  \ingroup buildblock
  \brief provides names for some numeric types and methods for finding their properties.

  \warning CHAR itself is missing (there's only signed and unsigned versions).
  This is because this class is only used where you have to know if the type
  is signed.
  */


class NumericType
{
public:
  //! An enum for the supported types. 
  // BIT has to be the first type, and UNKNOWN_TYPE has to be 
  // the last name for the implementation of the 2nd NumericType constructor 
  // to work
  enum Type { BIT, SCHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, 
	 FLOAT, DOUBLE, UNKNOWN_TYPE };

  //! stores the current type
  Type id;

  //! constructor, defaulting to unknown type
  inline NumericType(Type t = UNKNOWN_TYPE);

  //! A constructor to work from named types a la Interfile
  /*!
   Possible values for \a number_format are 
   <tt>bit</tt>, <tt>signed integer</tt>, <tt>unsigned integer</tt>, <tt>float</tt>
   Exact types are determined via the size_in_bytes parameter.
  */
  NumericType(const string& number_format, const size_t size_in_bytes);

  //! comparison operator
  inline bool operator==(NumericType type) const;
  inline bool operator!=(NumericType type) const;

  //! as reported by sizeof(), so it is really size_in_sizeof_char
  size_t size_in_bytes() const;
  
  size_t size_in_bits() const;

  //! returns true for all built-in types, except \c unsigned types
  bool signed_type() const;

  //! returns true for all built-in types, except \c float and \c double
  bool integer_type() const;


  //! returns the names and size a la Interfile. see NumericType(const string&,const size_t)
  void get_Interfile_info(string& number_format, size_t& size_in_bytes) const;
};

/*!
  \ingroup buildblock
  \brief A helper class that specifies the C++ type for a particular NumericType.

  Use as follows:
  \code
  typedef TypeForNumericType<NumericType::SHORT>::type current_type;
  \endcode
  This is useful when writing code depending on the value of a NumericType enum.
*/
template <int numeric_type_enum>
struct TypeForNumericType {};


#ifndef DOXYGEN_SKIP // disable definitions when running doxygen to avoid having all of this in the doc
template <>
struct TypeForNumericType<NumericType::SCHAR> 
{ typedef signed char type; };
template <>
struct TypeForNumericType<NumericType::UCHAR> 
{ typedef unsigned char type; };
template <>
struct TypeForNumericType<NumericType::SHORT> 
{ typedef signed short type; };
template <>
struct TypeForNumericType<NumericType::USHORT> 
{ typedef unsigned short type; };
template <>
struct TypeForNumericType<NumericType::INT> 
{ typedef signed int type; };
template <>
struct TypeForNumericType<NumericType::UINT> 
{ typedef unsigned int type; };
template <>
struct TypeForNumericType<NumericType::LONG> 
{ typedef signed long type; };
template <>
struct TypeForNumericType<NumericType::ULONG> 
{ typedef unsigned long type; };
template <>
struct TypeForNumericType<NumericType::FLOAT> 
{ typedef float type; };
template <>
struct TypeForNumericType<NumericType::DOUBLE> 
{ typedef double type; };
#endif

END_NAMESPACE_STIR

#include "stir/NumericType.inl"

#endif
