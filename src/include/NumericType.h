//
//  $Id$: $Date$
//

#ifndef __NumericType_H__
#define __NumericType_H__

/*!
  \file 
  \ingroup buildblock 
  \brief This file declares the NumericType class.

  \author Kris Thielemans 
  \author PARAPET project

  \date    $Date$

  \version $Revision$
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

*/

#include "Tomography_common.h"
#include <string>

START_NAMESPACE_TOMO

#ifndef TOMO_NO_NAMESPACES
using std::size_t;
using std::string;
#endif

/*!
  \ingroup buildblock
  \brief 
  provides names for some numeric types and methods for finding their properties.

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
  // TODOdoc
  // Possible names are "bit, "signed integer", "signed integer", "float"
  // Exact types are determined via the size_in_bytes parameter.
  NumericType(const string number_format, const size_t size_in_bytes);

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


  //! returns the names and size a la Interfile. see NumericType(const string,const size_t)
  void get_Interfile_info(string& number_format, size_t& size_in_bytes) const;
};

END_NAMESPACE_TOMO

#include "NumericType.inl"

#endif
