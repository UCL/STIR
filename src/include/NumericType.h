//
// $Id$: $Date$
//

#ifndef __NumericType_H__
#define __NumericType_H__

/*
  class NumericType provides names for some numeric types and some properties.

  History : 
  first version by Kris Thielemans

  KT 28/07/98 
  added two methods to NumericType:
  inline bool signed_type() const;
  inline bool integer_type() const;
  
  KT 15/12/99
  removed inlining. Gave troubles with gcc on NT.
  added method that returns Interfile info

*/

// for namespace std
#include "pet_common.h"
// (for NumericType constructor)
#include <string>

class NumericType
{
public:
  // Names for the supported types. 
  // BIT has to be the first type, and UNKNOWN_TYPE has to be 
  // the last name for the implementation of the 2nd NumericType constructor 
  // to work
  enum Type { BIT, SCHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, 
	 FLOAT, DOUBLE, UNKNOWN_TYPE };

  Type id;

  inline NumericType(Type t = UNKNOWN_TYPE);

  // A constructor to work from named types a la Interfile
  // Possible names are "bit, "signed integer", "signed integer", "float"
  // Exact types are determined via the size_in_bytes parameter.
  // KT 15/12/99 renamed last parameter, and use type size_t instead of int
  NumericType(const string number_format, const size_t size_in_bytes);

  inline bool operator==(NumericType type) const;

  // as reported by sizeof(), so really size_in_sizeof_char
  size_t size_in_bytes() const;
  size_t size_in_bits() const;
  bool signed_type() const;
  bool integer_type() const;


  // KT 15/12/99 new
  // This will return the names and size a la Interfile (see above)
  void get_Interfile_info(string& number_format, size_t& size_in_bytes) const;
};

#include "NumericType.inl"

#endif
