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

  History : 
  first version by KT
  KT 28/07/98 
  added two methods to NumericType:
  inline bool signed_type() const;
  inline bool integer_type() const;
  KT 01/08/98 
  added class ByteOrder
  added constructor for NumericType to work with names
 */ 

// for PETerror
#include "pet_common.h"
// for swap()
#include <algorithm>
// KT 01/08/98 new (for new NumericType constructor)
#include <string>

// KT this code uses' #define's  as specified by the ANSI C standard
#include <limits.h>
#include <float.h>
#ifndef __MSL__
#include <sys/types.h>
#else
#include <size_t.h>
#endif


// KT 01/08/98 added 
class ByteOrder
{
public: 
  // 'big_endian' is like x86, MIPS
  // 'little_endian' is like PowerPC, Sparc
  // 'native' means the same order as the machine the program is running on
  // 'swapped' means the opposite order
  enum Order { little_endian, big_endian, native, swapped };

  // static methods (i.e. not using 'this')

  inline static Order get_native_order()
  { 
    // for efficiency, this refers to a static member.
    return native_order;
  }

  // Note: this uses member templates. If your compiler cannot handle it,
  // you will have to write lots of lines for the built-in types
  template <class NUMBER>
  void static swap_order(NUMBER& a)
    { 
      switch (sizeof(NUMBER))
      {
      case 1:
	break;
      case 2:
	swaporder2((char *) &a);
	break;
      case 4:
	swaporder4((char *) &a);
	break;
      case 8:
	swaporder8((char *) &a);
	break;
      default:
	PETerror("ByteOrder::swaporder called with sizeof(type) other than 1,2,4,8");
	Abort();
      }
    }

  // constructor, defaulting to 'native' byte order
  ByteOrder(Order byte_order = native)
    : byte_order(byte_order)
  {}

  bool operator==(ByteOrder order2) const
  {
    return byte_order == order2.byte_order;
  }

  inline bool is_native_order() const
  {
    return (byte_order == native ||
	    byte_order == get_native_order());
  }

  template <class NUMBER>
  void swap_if_necessary(NUMBER& a) const
  {
     if (!is_native_order)
       ByteOrder::swap_order(a);
  }

private:
  // This static member has to be initialised somewhere (in file scope).
  const static Order native_order ;

  Order byte_order;
 
  void static swaporder2(char * a) 
   { swap(a[0], a[1]);  }

   void static swaporder4(char * a) 
   { swap(a[0], a[3]); swap(a[1], a[2]); }

   void static swaporder8(char * a) 
   { swap(a[0], a[7]); swap(a[1], a[6]); swap(a[2], a[5]); swap(a[3], a[4]); }
    
};

class NumericType
{
public:
  // Names for the supported types. 
  // BIT has to be the first type, and UNKNOWN_TYPE has to be 
  // the last name for the implementation of the 2nd NumericType constructor 
  // to work
  enum Type { BIT, SCHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, 
	 FLOAT, DOUBLE, UNKNOWN_TYPE } id;
  // KT 06/05/98 added default value
  NumericType(Type t = UNKNOWN_TYPE)
    : id(t) 
    {}

  bool operator==(NumericType type) const
    { return id == type.id; }
  // as reported by sizeof(), so really size_in_sizeof_char
  // implementation is at the end of the file, as it uses NumericInfo below
  inline size_t size_in_bytes() const;
  size_t size_in_bits() const
    { return CHAR_BIT * size_in_bytes(); }
  // KT 28/07/98 new
  inline bool signed_type() const;
  inline bool integer_type() const;

  // KT 01/98/98 new
  // A constructor to work from named types a la Interfile
  // Possible names are "bit, "signed integer", "signed integer", "float"
  // Exact types are determined via the bytes_per_pixel parameter.
  NumericType(const string number_format, const int bytes_per_pixel);

};

// the general framework of NumericInfo
// This declaration is essentially empty, but needed to use template syntax.
// It could be better to have some error in here, flagging (at compile time)
// when using it for a type which is not yet in our list. With the code 
// below, you'll see a linker error only, something like
// 'undefined reference NumericInfo<complex>::signed_type()'
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

// KT 09/08/98 completed list of possible values
inline size_t NumericType::size_in_bytes() const
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

// KT 28/07/98 new
// KT 09/08/98 completed list of possible values
inline bool NumericType::signed_type() const
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

// KT 09/08/98 completed list of possible values
inline bool NumericType::integer_type() const
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

#endif
