// $Id$

/*!
  \file 
  \ingroup Array 
  \brief non-inline implementations for the Array class 

  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$

  Currently, this file needs to contain only read_data, write_data.

  The end of this file contains instantiations for some
  common cases. If you have linking problems with non-inline Array member 
  functions, you might need to add your own types in the list.

  For compilers that do not support partial template specialisation,
  the 1D implementations are rather tedious: full specialisations
  for a few common types. Result: lots of code repetition.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/convert_array.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
void 
Array<num_dimensions, elemT>::grow(const IndexRange<num_dimensions>& range)
{
  base_type::grow(range.get_min_index(), range.get_max_index());
  base_type::iterator iter = base_type::begin();
  IndexRange<num_dimensions>::const_iterator range_iter = range.begin();
  for (;
       iter != base_type::end(); 
       iter++, range_iter++)
    (*iter).grow(*range_iter);

  is_regular_range = range.is_regular();
}

/*! When the \c type parameter matches \c elemT, \c scale will always be set to 1. */
template <int num_dimensions, class elemT>
void 
Array<num_dimensions, elemT>::read_data(istream& s, NumericType type, float& scale,
				 const ByteOrder byte_order)
{
  base_type::check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO? you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = float(1);
      return;
    }
#define read_and_convert(type) \
        Array<num_dimensions,type> data(get_index_range()); \
	data.read_data(s, byte_order); \
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
  switch(type.id)
    {
    case NumericType::SHORT:
      {
	read_and_convert(signed short);	
	break;
      }
    case NumericType::USHORT:
      {
	read_and_convert(unsigned short);
	break;
      }
    case NumericType::FLOAT:
      {
	read_and_convert(float);
	break;
      }
    default:
      error("Array::read_data : type not yet supported\n, edit line %d in file %s",
	       __LINE__, __FILE__);    
  }
#undef read_and_convert

  base_type::check_state();
}

/**************************** write_data *****************************/

/*
  Current implementation simply calls convert() and writes its result.

  An alternative implementation would be to find the scale factor first,
  and then call write_data for its members with that scale factor.
  The problem would be how to tell the members that they don't need
  to find the scale factor again. I think this would require an extra 
  (private) function 'write_data_fixed_scaling_factor'.
  It would however save us allocating a temporary multi-dimensional 
  tensor.
*/

#ifndef MEMBER_TEMPLATES

/*! When the \c type parameter matches \c elemT, \c scale will always be set to 1. */
template <int num_dimensions, class elemT>
void 
Array<num_dimensions, elemT>::write_data(ostream& s, NumericType type, 
		                         float& scale,
		                         const ByteOrder byte_order ) const
{
  base_type::check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      write_data(s, byte_order);
      // you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = float(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
#define convert_and_write(type) \
	Array<num_dimensions,type> data = \
	  convert_array(scale, *this,  NumericInfo<type>()); \
	data.write_data(s, byte_order);

  switch(type.id)
    {
    case NumericType::SHORT:
      {
	convert_and_write(signed short);	
	break;
      }
    case NumericType::USHORT:
      {
	convert_and_write(unsigned short);
	break;
      }
    case NumericType::FLOAT:
      {
	convert_and_write(float);
	break;
      }
    default:
      error("Array::write_data : type not yet supported\n, edit line %d in file %s",
	       __LINE__, __FILE__);
    }
#undef convert_and_write

  base_type::check_state();
}

#else // MEMBER_TEMPLATES

/*! When \c elemT is equal to \c elemT2, \c scale will always be set to 1. */
template <int num_dimensions, class elemT, class elemT2, class scaleT>
void Array<num_dimensions,elemT>::write_data(ostream& s, 
				  NumericInfo<elemT2> info2, 
				  scaleT& scale,
				  const ByteOrder byte_order) const
{
  base_type::check_state();
  Array<num_dimensions,elemT2> data = 
    convert_array(scale, *this,  NumericInfo<type>());
  data.write_data(s, byte_order);
}

/*! Partial specialisation to \c elemT = \c elemT2, \c scale will always be set to 1. */
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dimensions, class elemT, class scaleT>
void 
Array<num_dimensions,elemT>::write_data(ostream& s, 
				  NumericInfo<elemT> info2, 
				  scaleT& scale,
				  const ByteOrder byte_order) const
{
  base_type::check_state();
  data.write_data(s, byte_order);
  // TODO you might want to use the scale even in this case, 
  // but at the moment we don't
  scale = scaleT(1);
}
#endif

/*! When the \c type parameter matches \c elemT, \c scale will always be set to 1. */
template <int num_dimensions, class elemT>
void 
Array<num_dimensions,elemT>::write_data(ostream& s, 
				  NumericType type, 
				  float& scale,
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
    case NumericType::SHORT:
      {
	typedef signed short type;
	data.write_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	data.write_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	data.write_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    default:
      error("Array::write_data : type not yet supported\n, edit line %d in file %s",
	       __LINE__, __FILE__); 
    }
}

#endif // MEMBER_TEMPLATES


/*************************************
 1D specialisation
 *************************************/

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/************************** 2 arg version *******************************/

template <class elemT>
void
Array<1, elemT>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  base_type::check_state();
  
  if (!s)
  { error("Array::read_data: error before reading from stream\n"); }
  if (s.eof())
  { error("Array::read_data: reading past EOF\n"); }
  
  s.read(reinterpret_cast<char *>(base_type::get_data_ptr()), base_type::length * sizeof(elemT));
  base_type::release_data_ptr();
  
  if (!s)
  { error("Array::read_data: error after reading from stream \n"); }
  base_type::check_state();
  
  if (!byte_order.is_native_order())
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
    
}

template <class elemT>
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
  s.write(reinterpret_cast<const char *>(base_type::begin()), base_type::length * sizeof(elemT));   
  
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

#ifndef  MEMBER_TEMPLATES

/*! When the \c type parameter matches \c elemT, \c scale will always be set to 1. */
template <class elemT>
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
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data(base_type::get_min_index(), base_type::get_max_index());
	data.read_data(s, byte_order);
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data(base_type::get_min_index(), base_type::get_max_index());
	data.read_data(s, byte_order);
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data(base_type::get_min_index(), base_type::get_max_index());
	data.read_data(s, byte_order);
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );
	break;
      }
    default:
      // TODO
      error("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      
    }

  base_type::check_state();
}

// KT 17/05/2000 forgot to add write_data...

/*! When the \c type parameter matches \c elemT, \c scale will always be set to 1. */
template <class elemT>
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
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data = 
	  convert_array(scale, *this, NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data = 
	  convert_array(scale, *this, NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data = 
	  convert_array(scale, *this, NumericInfo<type>());
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

#else // MEMBER_TEMPLATES

#error This code has never been checked. It might work, but maybe not...

/********************** read **********************/

/*! When \c elemT is equal to \c elemT2, \c scale will always be set to 1. */
template <class elemT, class elemT2, class scaleT>
void Array<1,elemT>::read_data(istream& s, 
				 NumericInfo<elemT2> info2, 
				 scaleT& scale,
				 const ByteOrder byte_order)
{
  base_type::check_state();
  Array<1,elemT2> data(base_type::get_min_index(), base_type::get_max_index());
  data.read_data(s, byte_order);
  operator=( convert_array(scale, data, NumericInfo<elemT>()) );
  base_type::check_state();
}

/*! Partial specialisation to \c elemT = \c elemT2, \c scale will always be set to 1. */
template <class elemT, class scaleT>
void Array<1,elemT>::read_data(istream& s,
				 NumericInfo<elemT> info2, 
				 scaleT& scale,
				 const ByteOrder byte_order)
{
  base_type::check_state();
  data.read_data(s, byte_order);
  // TODO you might want to use the scale even in this case, 
  // but at the moment we don't
  scale = scaleT(1);
  base_type::check_state();
}


template <class elemT>
void Array<1,elemT>::read_data(ostream& s, 
				  NumericType type, 
				  float& scale,
				  const ByteOrder byte_order) const
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
    case NumericType::SHORT:
      {
	typedef signed short type;
	data.read_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	data.read_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	data.read_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    default:
      // TODO
      error("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
    }
}

/********************** write **********************/
// KT 17/05/2000 forgot to add write_data...


/*! When \c elemT is equal to \c elemT2, \c scale will always be set to 1. */
template <class elemT, class elemT2, class scaleT>
void Array<1,elemT>::write_data(ostream& s, 
				 NumericInfo<elemT2> info2, 
				 scaleT& scale,
				 const ByteOrder byte_order) const
{
  base_type::check_state();
  Array<elemT2> data = 
    convert_array(scale, *this, NumericInfo<elemT2>());
  data.write_data(s, byte_order);
}

/*! Partial specialisation to \c elemT = \c elemT2, \c scale will always be set to 1. */
template <class elemT, class scaleT>
void Array<1,elemT>::write_data(ostream& s,
				 NumericInfo<elemT> info2, 
				 scaleT& scale,
				 const ByteOrder byte_order) const
{
  base_type::check_state();
  data.write_data(s, byte_order);
  // TODO you might want to use the scale even in this case, 
  // but at the moment we don't
  scale = scaleT(1);
}

/*! When the \c type parameter matches \c elemT, \c scale will always be set to 1. */
template <class elemT>
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
    case NumericType::SHORT:
      {
	typedef signed short type;
	data.write_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	data.write_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	data.write_data(s, NumericInfo<type>(), scale, byte_order);
	break;
      }
    default:
      // TODO
      error("Array::write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      
    }

  base_type::check_state();
}

#endif // MEMBER_TEMPLATES


#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/************************** 2 arg version *******************************/

/************** float ******************/

void
Array<1, float>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  base_type::check_state();
  
  if (!s)
  { error("Error before reading from stream in read_data\n");  }
  if (s.eof())
  { error("Reading past EOF in read_data\n");  }
  
  s.read(reinterpret_cast<char *>(base_type::get_data_ptr()), base_type::length * sizeof(elemT));
  base_type::release_data_ptr();
  
  if (!s)
  { error("Error after reading from stream in read_data\n");  }
  base_type::check_state();
  
  if (!byte_order.is_native_order())
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
    
}

void
Array<1, float>::write_data(ostream& s,
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
  s.write(reinterpret_cast<const char *>(base_type::begin()), base_type::length * sizeof(elemT));   
  
  if (!s)
  { error("Array::write_data: error after writing to stream.\n");  }
  
  if (!byte_order.is_native_order())
  {
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
  }
  
  base_type::check_state();
}

/************** short ******************/

void
Array<1, short>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  base_type::check_state();
  
  if (!s)
  { error("Error before reading from stream in read_data\n");  }
  if (s.eof())
  { error("Reading past EOF in read_data\n");  }
  
  s.read(reinterpret_cast<char *>(base_type::get_data_ptr()), base_type::length * sizeof(elemT));
  base_type::release_data_ptr();
  
  if (!s)
  { error("Error after reading from stream in read_data\n");  }
  base_type::check_state();
  
  if (!byte_order.is_native_order())
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
    
}

void
Array<1, short>::write_data(ostream& s,
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
  s.write(reinterpret_cast<const char *>(base_type::begin()), base_type::length * sizeof(elemT));   
  
  if (!s)
  { error("Array::write_data: error after writing to stream.\n");  }
  
  if (!byte_order.is_native_order())
  {
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
  }
  
  base_type::check_state();

}/************** unsigned short ******************/


void
Array<1, unsigned short>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  base_type::check_state();
  
  if (!s)
  { error("Error before reading from stream in read_data\n");  }
  if (s.eof())
  { error("Reading past EOF in read_data\n");  }
  
  s.read(reinterpret_cast<char *>(base_type::get_data_ptr()), base_type::length * sizeof(elemT));
  base_type::release_data_ptr();
  
  if (!s)
  { error("Error after reading from stream in read_data\n");  }
  base_type::check_state();
  
  if (!byte_order.is_native_order())
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
    
}

void
Array<1, unsigned short>::write_data(ostream& s,
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
  s.write(reinterpret_cast<const char *>(base_type::begin()), base_type::length * sizeof(elemT));   
  
  if (!s)
  { error("Array::write_data: error after writing to stream.\n");  }
  
  if (!byte_order.is_native_order())
  {
    for(int i=base_type::get_min_index(); i<=base_type::get_max_index(); i++)
      ByteOrder::swap_order(base_type::num[i]);
  }
  
  base_type::check_state();
}

#if !defined(_MSC_VER) || (_MSC_VER > 1100)

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
/******************* float *********************/

#define elemT float

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
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data = 
	  CONVERT_ARRAY_WRITE(scale, *this, NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data = 
	  CONVERT_ARRAY_WRITE(scale, *this, NumericInfo<type>());
	data.write_data(s, byte_order);
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data = 
	  CONVERT_ARRAY_WRITE(scale, *this, NumericInfo<type>());
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

#undef elemT



/******************** short ********************/

#define elemT short

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

#undef elemT


/************** unsigned short ******************/

#define elemT unsigned short

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

#undef elemT

#endif  // VC 5

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/**************************************************
 instantiations
 **************************************************/

// add any other types you need
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template class Array<1,signed char>;
template class Array<1,short>;
template class Array<1,unsigned short>;
template class Array<1,float>;
#endif

#if !defined(_MSC_VER) || _MSC_VER>=1300
template class Array<2,signed char>;
#endif
template class Array<2,short>;
template class Array<2,unsigned short>;
template class Array<2,float>;

#if !defined(_MSC_VER) || _MSC_VER>=1300
template class Array<3, signed char>;
#endif
template class Array<3, short>;
template class Array<3,unsigned short>;
template class Array<3,float>;


template class Array<4, short>;
template class Array<4,unsigned short>;
template class Array<4,float>;


END_NAMESPACE_STIR
