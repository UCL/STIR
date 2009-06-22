//
// $Id$
//
#ifndef __stir_ByteOrder_H__
#define __stir_ByteOrder_H__

/*!
  \file 
  \ingroup buildblock 
  \brief This file declares the stir::ByteOrder class.

  \author Kris Thielemans 
  \author Alexey Zverovich
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
/*
  Modification History:

  - First version by KT

  - AZ&KT 15/12/99 rewrote swap_order using revert_region
*/

#include "stir/common.h"
// for swap
#include <algorithm>

START_NAMESPACE_STIR

/*!
  \class ByteOrder
  \ingroup buildblock
  \brief This class provides member functions to 
  find out what byte-order your machine is and to swap numbers.

  \par Some machine architectures:
  -Little endian processors: Intel, Alpha, VAX, PDP-11, ...
  -Big endian processors: Sparc, PowerPC, Motorola 68???, ...

  In a little-endian architecture, within a given 16- or 32-bit word, 
   bytes at lower addresses have lower significance 
   (the word is stored `little-end-first').
   (Quoted from http://www.techfak.uni-bielefeld.de/~joern/jargon/)
*/

/*
  \brief Internal class to swap bytes.

  This defines a revert function that swaps the outer elements of an
  array, and then calls itself recursively. 
  Due to the use of templates, the recursion gets resolved at compile 
  time.
  (We have to use a class for this, as we rely on
  template specialisation, which is unavailable for functions).
    
  \warning This class should not be used anywhere, except in the 
  ByteOrder::swap_order implementation.

  \internal
*/
/* 
  It really shouldn't be in this include file, but we have to because
  of a compiler bug in VC++ (member templates have to be defined in the
  class).
*/
//TODO put revert_region in own namespace or as private class of ByteOrder


template <int size>
class revert_region
{
public:
  inline static void revert(unsigned char* ptr)
  {
#ifndef STIR_NO_NAMESPACES
    std::swap(ptr[0], ptr[size - 1]);
#else
    swap(ptr[0], ptr[size - 1]);
#endif
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
  //! enum for specifying the byte-order
  enum Order {
    little_endian, /*!< is like x86, MIPS*/
    big_endian, /*!< is like PowerPC, Sparc.*/
    native, /*!< means the same order as the machine the programme is running on*/
    swapped /*!<  means the opposite order of \e native*/
  };

  //******* static members (i.e. not using 'this')

  //! returns the byte-order native to the machine the programme is running on.
  inline static Order get_native_order();

  // Note: this uses member templates. If your compiler cannot handle it,
  // you will have to write lots of lines for the built-in types
  // Note: Implementation has to be inside the class definition to get 
  // this compiled by VC++ 5.0 and 6.0
  //! swap the byteorder of the argument
  template <class NUMBER>
  inline static void swap_order(NUMBER& value)
  {
	revert_region<sizeof(NUMBER)>::revert(reinterpret_cast<unsigned char*>(&value));
  }

  //********* non-static members

  //! constructor, defaulting to 'native' byte order
  inline ByteOrder(Order byte_order = native);

  //! comparison operator
  inline bool operator==(const ByteOrder order2) const;
  inline bool operator!=(const ByteOrder order2) const;

  //! check if the object refers to the native order.
  inline bool is_native_order() const;

  // Note: this uses member templates. If your compiler cannot handle it,
  // you will have to write lots of lines for the built-in types
  // Note: Implementation has to be inside the class definition to get 
  // this compiled by VC++ 5.0 and 6.0
  //! this swaps only when the order != native order
  template <class NUMBER> inline void swap_if_necessary(NUMBER& a) const
  {
    if (!is_native_order())
      swap_order(a);
  }


private:
  // This static member has to be initialised somewhere (in file scope).
  const static Order native_order ;

  Order byte_order;

};

END_NAMESPACE_STIR

#include "stir/ByteOrder.inl"

#endif
