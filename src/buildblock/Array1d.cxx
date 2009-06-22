
// $Id$
/*!
  \file 
  \ingroup Array 
  \brief implements the 1D specialisation of the stir::Array class for broken compilers

  \author Kris Thielemans (with help from Alexey Zverovich)
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

#if !defined(__stir_Array_H__) || !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) || !defined(elemT)
#error This file should only be included in Array.cxx for half-broken compilers
#endif

/* Lines here should really be identical to what you find as 1D specialisation 
   in convert_array.cxx, except that  template statements are dropped.
   */
/************************** 2 arg version *******************************/


void
Array<1, elemT>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  base_type::check_state();
  
  if (!s)
  { error("Error before reading from stream in read_data\n");  }
  if (s.eof())
  { error("Reading past EOF in read_data\n");  }
  
  const std::streamsize num_to_read =
    static_cast<std::streamsize>(base_type::size())* sizeof(elemT);
  s.read(reinterpret_cast<char *>(base_type::get_data_ptr()), num_to_read);
  base_type::release_data_ptr();
  
  if (!s)
  { error("Error after reading from stream in read_data\n");  }
  base_type::check_state();
  
  if (!byte_order.is_native_order())
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
    
}

void
Array<1, elemT>::write_data(ostream& s,
			    const ByteOrder byte_order) const
{
  base_type::check_state();
  
  // TODO handling of byte-swapping is unsafe when we wouldn't call abort()
  // While writing, the tensor is byte-swapped.
  // Safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<1, T> a_copy(*this);
  for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
  ByteOrder::swap_order(a_copy[i]);
  a_copy.write_data(s);
  return;
  }
  */
  if (!byte_order.is_native_order())
  {
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
  }
  
  if (!s)
  { error("Array::write_data: error before writing to stream.\n");  }
  
  // TODO use get_const_data_ptr() when it's a const member function
  s.write(reinterpret_cast<const char *>(base_type::begin()), base_type::get_length() * sizeof(elemT));   
  
  if (!s)
  { error("Array::write_data: error after writing to stream.\n");  }
  
  if (!byte_order.is_native_order())
  {
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
  }
  
  base_type::check_state();
}

/************************** 4 arg version *******************************/
/*
 VC 5.0 has a bug that it cannot resolve the num_dimensions template-arg
 when using convert_array. You have to specify all template args explicitly.
 I do this here with macros to prevent other compilers breaking on it
 (notably VC 6.0...)
 These macros depend on the exact lines used below. Sorry.
 */

#if defined(_MSC_VER) && (_MSC_VER < 1200)
#define CONVERT_ARRAY_WRITE convert_array<1,elemT,type,float>
#define CONVERT_ARRAY_READ convert_array<1,type,elemT,float>
#else
#define CONVERT_ARRAY_WRITE convert_array
#define CONVERT_ARRAY_READ convert_array
#endif

/* following are copies of the template definitions above except
  - there's a work-around for the VC 5.00 bug
  - the template <class elemT> lines are deleted, and replaced
    by #define elemT sometype, #undef elemT pairs
*/

void Array<1,elemT>::write_data(ostream& s, NumericType type, float& scale,
				  const ByteOrder byte_order) const
{
  base_type::check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      write_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = float(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
  switch(type.id)
    {
#if !defined(_MSC_VER) || _MSC_VER>=1300
    case NumericType::SCHAR:
      {
	typedef signed char type;
	Array<1,type> data = 
	  CONVERT_ARRAY_WRITE(scale, *this,  NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
#endif
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data = 
	  CONVERT_ARRAY_WRITE(scale, *this,  NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data = 
	  CONVERT_ARRAY_WRITE(scale, *this,  NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data = 
	  CONVERT_ARRAY_WRITE(scale, *this,  NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
    default:
      // TODO
      error("Array::write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      
    }

  base_type::check_state();
}

void Array<1,elemT>::read_data(istream& s, NumericType type, float& scale,
				 const ByteOrder byte_order)
{
  base_type::check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = float(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
  switch(type.id)
    {
#if !defined(_MSC_VER) || _MSC_VER>=1300
    case NumericType::SCHAR:
      {
	typedef signed char type;
	Array<1,type> data(base_type::get_min_index(), base_type::get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
#endif
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data(base_type::get_min_index(), base_type::get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data(base_type::get_min_index(), base_type::get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data(base_type::get_min_index(), base_type::get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    default:
      // TODO
      error("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      
    }

  base_type::check_state();
}

#undef CONVERT_ARRAY_WRITE
#undef CONVERT_ARRAY_READ

#undef elemT

