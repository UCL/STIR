// $Id$: $Date$


/* implementation of Array::read_data and write_data
   first version : KT
   */
#include "convert_array.h"

START_NAMESPACE_TOMO

template <int num_dimensions, class elemT>
void 
Array<num_dimensions, elemT>::read_data(istream& s, NumericType type, Real& scale,
				 const ByteOrder byte_order)
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);
      return;
    }
/*
 VC 5.0 has a bug that it cannot resolve the num_dimensions template-arg
 when using convert_array. You have to specify all template args explicitly.
 I do this here with macros to prevent other compilers breaking on it
 (notably VC 6.0...)
 */
#if defined(_MSC_VER) && (_MSC_VER < 1200)
#define read_and_convert(type) \
        Array<num_dimensions,type> data(get_index_range()); \
	data.read_data(s, byte_order); \
	operator=( convert_array<num_dimensions,type,elemT,Real>(scale, data, NumericInfo<elemT>()) );
#else
#define read_and_convert(type) \
        Array<num_dimensions,type> data(get_index_range()); \
	data.read_data(s, byte_order); \
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );
#endif

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
      // TODO
      PETerror("type not yet supported"); Abort();
    }
#undef read_and_convert

  check_state();
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

template <int num_dimensions, class elemT>
void 
Array<num_dimensions, elemT>::write_data(ostream& s, NumericType type, 
		                         Real& scale,
		                         const ByteOrder byte_order ) const
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      write_data(s, byte_order);
      // you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
/*
 VC 5.0 has a bug that it cannot resolve the num_dimensions template-arg
 when using convert_array. You have to specify all template args explicitly.
 I do this here with macros to prevent other compilers breaking on it
 (notably VC 6.0...)
 */
#if defined(_MSC_VER) && (_MSC_VER < 1200)
#define convert_and_write(type) \
	Array<num_dimensions,type> data = \
	  convert_array<num_dimensions,elemT,type,Real>(scale, *this,  NumericInfo<type>()); \
	data.write_data(s, byte_order);
#else
#define convert_and_write(type) \
	Array<num_dimensions,type> data = \
	  convert_array(scale, *this,  NumericInfo<type>()); \
	data.write_data(s, byte_order);
#endif

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
      // TODO
      PETerror("type not yet supported"); Abort();
    }
#undef convert_and_write

  check_state();
}

#else // MEMBER_TEMPLATES

template <int num_dimensions, class elemT, class elemT2, class scaleT>
void Array<num_dimensions,elemT>::write_data(ostream& s, 
				  NumericInfo<elemT2> info2, 
				  scaleT& scale,
				  const ByteOrder byte_order) const
{
  check_state();
  Array<num_dimensions,elemT2> data = 
    convert_array(scale, *this,  NumericInfo<type>());
  data.write_data(s, byte_order);
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dimensions, class elemT, class scaleT>
void 
Array<num_dimensions,elemT>::write_data(ostream& s, 
				  NumericInfo<elemT> info2, 
				  scaleT& scale,
				  const ByteOrder byte_order) const
{
  check_state();
  data.write_data(s, byte_order);
  // TODO you might want to use the scale even in this case, 
  // but at the moment we don't
  scale = scaleT(1);
}
#endif

template <int num_dimensions, class elemT>
void 
Array<num_dimensions,elemT>::write_data(ostream& s, 
				  NumericType type, 
				  Real& scale,
				  const ByteOrder byte_order) const
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      write_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);
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
      PETerror("Array::write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); abort();
    }
}

#endif // MEMBER_TEMPLATES

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template <class elemT>
void
Array<1, elemT>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  check_state();
  
  if (!s)
  { PETerror("Error before reading from stream in read_data\n"); abort(); }
  if (s.eof())
  { PETerror("Reading past EOF in read_data\n"); abort(); }
  
  s.read(reinterpret_cast<char *>(get_data_ptr()), length * sizeof(elemT));
  release_data_ptr();
  
  if (!s)
  { PETerror("Error after reading from stream in read_data\n"); abort(); }
  check_state();
  
  if (!byte_order.is_native_order())
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
    
}

template <class elemT>
void
Array<1, elemT>::write_data(ostream& s,
			    const ByteOrder byte_order) const
{
  check_state();
  
  // TODO handling of byte-swapping is unsafe when we wouldn't call abort()
  // While writing, the tensor is byte-swapped.
  // Safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<1, T> a_copy(*this);
  for(int i=get_min_index(); i<=get_max_index(); i++)
  ByteOrder::swap_order(a_copy[i]);
  a_copy.write_data(s);
  return;
  }
  */
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  if (!s)
  { PETerror("Error before writing to stream in write_data\n"); abort(); }
  
  // TODO use get_const_data_ptr() when it's a const member function
  s.write(reinterpret_cast<const char *>(begin()), length * sizeof(elemT));   
  
  if (!s)
  { PETerror("Error after writing to stream in write_data\n"); abort(); }
  
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  check_state();
}


#ifndef  MEMBER_TEMPLATES

template <class elemT>
void Array<1,elemT>::read_data(istream& s, NumericType type, Real& scale,
				 const ByteOrder byte_order)
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
  switch(type.id)
    {
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( convert_array(scale, data, NumericInfo<elemT>()) );
	break;
      }
    default:
      // TODO
      PETerror("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      abort();
    }

  check_state();
}


#else // MEMBER_TEMPLATES

template <class elemT, class elemT2, class scaleT>
void Array<1,elemT>::read_data(istream& s, 
				 NumericInfo<elemT2> info2, 
				 scaleT& scale,
				 const ByteOrder byte_order)
{
  check_state();
  Array<1,elemT2> data(get_min_index(), get_max_index());
  data.read_data(s, byte_order);
  operator=( convert_array(scale, data, NumericInfo<elemT>()) );
  check_state();
}

// this requires partial template specialisation
template <class elemT, class scaleT>
void Array<1,elemT>::read_data(istream& s,
				 NumericInfo<elemT> info2, 
				 scaleT& scale,
				 const ByteOrder byte_order)
{
  check_state();
  data.read_data(s, byte_order);
  // TODO you might want to use the scale even in this case, 
  // but at the moment we don't
  scale = scaleT(1);
  check_state();
}


template <class elemT>
void Array<1,elemT>::read_data(ostream& s, 
				  NumericType type, 
				  Real& scale,
				  const ByteOrder byte_order) const
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);
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
      PETerror("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); abort();
    }
}

#endif // MEMBER_TEMPLATES


#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/************************** 2 arg version *******************************/

/************** float ******************/

void
Array<1, float>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  check_state();
  
  if (!s)
  { PETerror("Error before reading from stream in read_data\n"); abort(); }
  if (s.eof())
  { PETerror("Reading past EOF in read_data\n"); abort(); }
  
  s.read(reinterpret_cast<char *>(get_data_ptr()), length * sizeof(elemT));
  release_data_ptr();
  
  if (!s)
  { PETerror("Error after reading from stream in read_data\n"); abort(); }
  check_state();
  
  if (!byte_order.is_native_order())
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
    
}

void
Array<1, float>::write_data(ostream& s,
			    const ByteOrder byte_order) const
{
  check_state();
  
  // TODO handling of byte-swapping is unsafe when we wouldn't call abort()
  // While writing, the tensor is byte-swapped.
  // Safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<1, T> a_copy(*this);
  for(int i=get_min_index(); i<=get_max_index(); i++)
    ByteOrder::swap_order(a_copy[i]);
  a_copy.write_data(s);
  return;
  }
  */
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  if (!s)
  { PETerror("Error before writing to stream in write_data\n"); abort(); }
  
  // TODO use get_const_data_ptr() when it's a const member function
  s.write(reinterpret_cast<const char *>(begin()), length * sizeof(elemT));   
  
  if (!s)
  { PETerror("Error after writing to stream in write_data\n"); abort(); }
  
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  check_state();
}

/************** short ******************/

void
Array<1, short>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  check_state();
  
  if (!s)
  { PETerror("Error before reading from stream in read_data\n"); abort(); }
  if (s.eof())
  { PETerror("Reading past EOF in read_data\n"); abort(); }
  
  s.read(reinterpret_cast<char *>(get_data_ptr()), length * sizeof(elemT));
  release_data_ptr();
  
  if (!s)
  { PETerror("Error after reading from stream in read_data\n"); abort(); }
  check_state();
  
  if (!byte_order.is_native_order())
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
    
}

void
Array<1, short>::write_data(ostream& s,
			    const ByteOrder byte_order) const
{
  check_state();
  
  // TODO handling of byte-swapping is unsafe when we wouldn't call abort()
  // While writing, the tensor is byte-swapped.
  // Safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<1, T> a_copy(*this);
  for(int i=get_min_index(); i<=get_max_index(); i++)
  ByteOrder::swap_order(a_copy[i]);
  a_copy.write_data(s);
  return;
  }
  */
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  if (!s)
  { PETerror("Error before writing to stream in write_data\n"); abort(); }
  
  // TODO use get_const_data_ptr() when it's a const member function
  s.write(reinterpret_cast<const char *>(begin()), length * sizeof(elemT));   
  
  if (!s)
  { PETerror("Error after writing to stream in write_data\n"); abort(); }
  
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  check_state();

}/************** unsigned short ******************/


void
Array<1, unsigned short>::read_data(istream& s, 
			   const ByteOrder byte_order)
{
  check_state();
  
  if (!s)
  { PETerror("Error before reading from stream in read_data\n"); abort(); }
  if (s.eof())
  { PETerror("Reading past EOF in read_data\n"); abort(); }
  
  s.read(reinterpret_cast<char *>(get_data_ptr()), length * sizeof(elemT));
  release_data_ptr();
  
  if (!s)
  { PETerror("Error after reading from stream in read_data\n"); abort(); }
  check_state();
  
  if (!byte_order.is_native_order())
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
    
}

void
Array<1, unsigned short>::write_data(ostream& s,
			    const ByteOrder byte_order) const
{
  check_state();
  
  // TODO handling of byte-swapping is unsafe when we wouldn't call abort()
  // While writing, the tensor is byte-swapped.
  // Safe way: (but involves creating an extra copy of the data)
  /*
  if (!byte_order.is_native_order())
  {
  Array<1, T> a_copy(*this);
  for(int i=get_min_index(); i<=get_max_index(); i++)
  ByteOrder::swap_order(a_copy[i]);
  a_copy.write_data(s);
  return;
  }
  */
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  if (!s)
  { PETerror("Error before writing to stream in write_data\n"); abort(); }
  
  // TODO use get_const_data_ptr() when it's a const member function
  s.write(reinterpret_cast<const char *>(begin()), length * sizeof(elemT));   
  
  if (!s)
  { PETerror("Error after writing to stream in write_data\n"); abort(); }
  
  if (!byte_order.is_native_order())
  {
    for(int i=get_min_index(); i<=get_max_index(); i++)
      ByteOrder::swap_order(num[i]);
  }
  
  check_state();
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
#define CONVERT_ARRAY_WRITE convert_array<1,elemT,type,Real>
#define CONVERT_ARRAY_READ convert_array<1,type,elemT,Real>
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

void Array<1,elemT>::write_data(ostream& s, NumericType type, Real& scale,
				  const ByteOrder byte_order) const
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      write_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

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
      PETerror("Array::write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      abort();
    }

  check_state();
}

void Array<1,elemT>::read_data(istream& s, NumericType type, Real& scale,
				 const ByteOrder byte_order)
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
  switch(type.id)
    {
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    default:
      // TODO
      PETerror("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      abort();
    }

  check_state();
}

#undef elemT



/******************** short ********************/

#define elemT short

void Array<1,elemT>::write_data(ostream& s, NumericType type, Real& scale,
				  const ByteOrder byte_order) const
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      write_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

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
      PETerror("Array::write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      abort();
    }

  check_state();
}

void Array<1,elemT>::read_data(istream& s, NumericType type, Real& scale,
				 const ByteOrder byte_order)
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
  switch(type.id)
    {
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    default:
      // TODO
      PETerror("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      abort();
    }

  check_state();
}

#undef elemT


/************** unsigned short ******************/

#define elemT unsigned short

void Array<1,elemT>::write_data(ostream& s, NumericType type, Real& scale,
				  const ByteOrder byte_order) const
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      write_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

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
      PETerror("Array::write_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      abort();
    }

  check_state();
}

void Array<1,elemT>::read_data(istream& s, NumericType type, Real& scale,
				 const ByteOrder byte_order)
{
  check_state();
  if (NumericInfo<elemT>().type_id() == type)
    {
      read_data(s, byte_order);
      // TODO you might want to use the scale even in this case, 
      // but at the moment we don't
      scale = Real(1);

      return;
    }

  // KT TODO pretty awful way of doings things, but I'm not sure how to handle it
  switch(type.id)
    {
    case NumericType::SHORT:
      {
	typedef signed short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::USHORT:
      {
	typedef unsigned short type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    case NumericType::FLOAT:
      {
	typedef float type;
	Array<1,type> data(get_min_index(), get_max_index());
	data.read_data(s, byte_order);
	operator=( CONVERT_ARRAY_READ(scale, data, NumericInfo<elemT>()) );
	break;
      }
    default:
      // TODO
      PETerror("Array::read_data : type not yet supported\n, at line %d in file %s",
	       __LINE__, __FILE__); 
      abort();
    }

  check_state();
}

#undef elemT

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/**************************************************
 instantiations
 **************************************************/

// add any other types you need
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template class Array<1,short>;
template class Array<1,unsigned short>;
template class Array<1,float>;
#endif

template class Array<2,short>;
template class Array<2,unsigned short>;
template class Array<2,float>;

template class Array<3, short>;
template class Array<3,unsigned short>;
template class Array<3,float>;

END_NAMESPACE_TOMO
