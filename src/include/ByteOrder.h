//
// $Id$: $Date$
//
#ifndef __ByteOrder_H__
#define __ByteOrder_H__

/*
   class ByteOrder provdes member functions to 
   - find out what byte order your machine is
   - swap numbers around

   In a little-endian architecture within a given 16- or 32-bit word, 
   bytes at lower addresses have lower significance 
   (the word is stored `little-end-first').
   (Quoted from http://www.techfak.uni-bielefeld.de/~joern/jargon/)

    Little endian processors: Intel, Alpha, VAX, PDP-11, ...
    Big endian processors: Sparc, PowerPC, Motorola 68???, ...


  History:
  First version by Kris Thielemans

  AZ&KT 15/12/99 moved inlines to separate file, rewrote swap_order
*/

/* 
  A class which should not be used anywhere else, except in the 
  swap_order implementation below.
  It really shouldn't be in this include file, but we have to because
  of a compiler bug in VC++ (member templates have to be defined in the
  class).

  This defines a revert function that swaps the outer elements of an
  array, and then calls itself recursively. 
  Due to the use of templates, the recursion gets resolved at compile 
  time.
*/
template <int size>
class revert_region
{
public:
  inline static void revert(unsigned char* ptr)
  {
    swap(ptr[0], ptr[size - 1]);
    revert_region<size - 2>::revert(ptr + 1);
  }
};

template <>
class revert_region<1>
{
public:
  inline static void revert(unsigned char* ptr)
  {}
};

template <>
class revert_region<0>
{
public:
  inline static void revert(unsigned char* ptr)
  {}
};


class ByteOrder
{
public: 
  // 'little_endian' is like x86, MIPS
  // 'big_endian' is like PowerPC, Sparc
  // 'native' means the same order as the machine the program is running on
  // 'swapped' means the opposite order
  enum Order { little_endian, big_endian, native, swapped };

  // static methods (i.e. not using 'this')

  inline static Order get_native_order();

  // Note: this uses member templates. If your compiler cannot handle it,
  // you will have to write lots of lines for the built-in types
  // Note: Implementation has to be inside the class definition to get 
  // this compiled by VC++ 5.0
  template <class NUMBER>
  inline static void swap_order(NUMBER& value)
  {
	revert_region<sizeof(NUMBER)>::revert(reinterpret_cast<unsigned char*>(&value));
  }

  // constructor, defaulting to 'native' byte order
  inline ByteOrder(Order byte_order = native);
  inline bool operator==(ByteOrder order2) const;

  inline bool is_native_order() const;

  // this swaps only when the order != native order
  template <class NUMBER> inline void swap_if_necessary(NUMBER& a) const;

private:
  // This static member has to be initialised somewhere (in file scope).
  const static Order native_order ;

  Order byte_order;

};

#include "ByteOrder.inl"

#endif
