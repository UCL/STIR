//
// $Id$: $Date$
//
/* Function to convert Array objects of different numeric types
Version 1.0: KT

 See convert_array.h for more information
 
*/

#include "convert_array.h"
#include <algorithm>
// for floor
#include <cmath>

START_NAMESPACE_TOMO

//! A local helper function to find that scale factor
template <class ArrayT, class T1, class T2, class scaleT>
void
find_scale_factor(const ArrayT& data_in, 
		  const NumericInfo<T1> info1, 
		  const NumericInfo<T2> info2, 
		  scaleT& scale_factor)
{
  if (info1.type_id() == info2.type_id())
  {
    // TODO could use different scale factor in this case as well, but at the moment we don't)
    scale_factor = scaleT(1);
    return;
  }

  // find the scale factor to use when converting to the maximum range in T2
  double tmp_scale = 
    static_cast<double>(data_in.find_max()) /  
    static_cast<double>(info2.max_value());
  if (info2.signed_type() && info1.signed_type())
  {
    tmp_scale = 
      max(tmp_scale, 
          static_cast<double>(data_in.find_min()) / 
	  static_cast<double>(info2.min_value()));
  }
  // use an extra factor of 1.01. Otherwise, rounding errors can
  // cause data_in.find_max() / scale_factor to be bigger than the 
  // max_value
  tmp_scale *= 1.01;
  
  if (scale_factor == 0 || tmp_scale > scale_factor)
  {
    // We need to convert to the maximum range in T2
    scale_factor = scaleT(tmp_scale);
  }
}

//! A local helper class in terms of which convert_array is defined
/*! Reason for this class is that we use partial template 
    specialisation (when supported by the compiler), and this
    is only possible with classes.
*/

// first for num_dimension > 1
template <int num_dimensions, class T1, class T2, class scaleT>
class convert_array_auxiliary
{
public:
  static inline void
    convert_array(Array<num_dimensions,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<num_dimensions,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<num_dimensions,T2>& data_out, 
       const scaleT scale_factor,
       const Array<num_dimensions,T1>& data_in)
  {
    Array<num_dimensions,T2>::iterator out_iter = data_out.begin();
    Array<num_dimensions,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {
      convert_array_auxiliary<num_dimensions-1,T1,T2,scaleT>::
        convert_array_fixed_scale_factor(*out_iter, scale_factor, *in_iter);
    }
  }
};


#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

// now for num_dimension == 1
template <class T1, class T2, class scaleT>
class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0      
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    Array<1,T2>::iterator out_iter = data_out.begin();
    Array<1,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

// special case T1=T2, num_dimensions>1
template <int num_dimensions, class T1, class scaleT>
class convert_array_auxiliary<num_dimensions,T1,T1,scaleT>
{
public:
  static inline void   
    convert_array(Array<num_dimensions,T1>& data_out, 
                  scaleT& scale_factor,
                  const Array<num_dimensions,T1>& data_in)
  {
    scale_factor = 1;
    data_out= data_in;
  }
};

// special case T1=T2, num_dimensions==1
// (needs to be here to sort out overloading of templates)
template <class T1, class scaleT>
class convert_array_auxiliary<1,T1,T1,scaleT>
{
public:
  static inline void
    convert_array(Array<1,T1>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    scale_factor = 1;
    data_out= data_in;
  }
};

#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

// we'll assume scaleT==Real
#define scaleT Real

/************** float, short *********************/
#define T1 float
#define T2 short

class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    Array<1,T2>::iterator out_iter = data_out.begin();
    Array<1,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

#undef T1
#undef T2

/************** float, unsigned short *********************/
#define T1 float
#define T2 unsigned short

class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    Array<1,T2>::iterator out_iter = data_out.begin();
    Array<1,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

#undef T1
#undef T2

/************** short, float *********************/
#define T1 short
#define T2 float

class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    Array<1,T2>::iterator out_iter = data_out.begin();
    Array<1,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

#undef T1
#undef T2

/************** short, unsigned short *********************/
#define T1 short
#define T2 unsigned short

class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    Array<1,T2>::iterator out_iter = data_out.begin();
    Array<1,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

#undef T1
#undef T2

/************** unsigned short, short *********************/
#define T1 unsigned short
#define T2 short

class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    Array<1,T2>::iterator out_iter = data_out.begin();
    Array<1,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

#undef T1
#undef T2

/************** unsigned short, float *********************/
#define T1 unsigned short
#define T2 float

class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    NumericInfo<T1> info1;
    NumericInfo<T2> info2;
    
    find_scale_factor(data_in, info1, info2, scale_factor);
    if (scale_factor == 0)
    {
      // data_in contains only 0, data_out is already 0
    }
    else
    {
      // do actual conversion
      convert_array_fixed_scale_factor(data_out, scale_factor, data_in);
    }
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T2>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    // KT: I coded the various checks on the data types in the loop.
    // This is presumably slow, but all these conditionals can be 
    // resolved at compile time, so a good compiler does the work for me.
    Array<1,T2>::iterator out_iter = data_out.begin();
    Array<1,T1>::const_iterator in_iter = data_in.begin();
    for (;
         in_iter != data_in.end();
         in_iter++, out_iter++)
    {    
      if (NumericInfo<T1>().signed_type() && !NumericInfo<T2>().signed_type() && *in_iter < 0)
      {
	// truncate negatives
	*out_iter = 0;
      }
      else
	*out_iter = 
 	  NumericInfo<T2>().integer_type() ?
	    static_cast<T2>(floor(*in_iter / scale_factor + 0.5)) :
            static_cast<T2>(*in_iter / scale_factor);
    }
  }
};

#undef T1
#undef T2

/******************* float, float ***************/
#define T1 float
class convert_array_auxiliary<1,T1,T1,scaleT>
{
public:
  static inline void
    convert_array(Array<1,T1>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    scale_factor = 1;
    data_out= data_in;
  }

  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T1>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    assert(scale_factor == 1);
    data_out= data_in;
  }
};
#undef T1

/******************* short, short ***************/
#define T1 short
class convert_array_auxiliary<1,T1,T1,scaleT>
{
public:
  static inline void
    convert_array(Array<1,T1>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    scale_factor = 1;
    data_out= data_in;
  }
  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T1>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    assert(scale_factor == 1);
    data_out= data_in;
  }
};
#undef T1

/******************* unsigned short, unsigned short ***************/
#define T1 unsigned short
class convert_array_auxiliary<1,T1,T1,scaleT>
{
public:
  static inline void
    convert_array(Array<1,T1>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    scale_factor = 1;
    data_out= data_in;
  }
  static inline void
    convert_array_fixed_scale_factor(
       Array<1,T1>& data_out, 
       const scaleT scale_factor,
       const Array<1,T1>& data_in)
  {
    assert(scale_factor == 1);
    data_out= data_in;
  }
};
#undef T1

#undef scaleT

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION


/***************************************************************
 Finally, the implementation of convert_array
***************************************************************/
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
Array<num_dimensions, T2>
convert_array(scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info2)
{
  Array<num_dimensions,T2> data_out(data_in.get_index_range());

  convert_array(data_out, scale_factor, data_in);
  return data_out;    
}
#endif
#if 1
template <int num_dimensions, class T1, class T2, class scaleT>
void 
convert_array(Array<num_dimensions, T2>& data_out,
	      scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in)
{
  convert_array_auxiliary<num_dimensions, T1, T2, scaleT>::
    convert_array(data_out, scale_factor, data_in);   
}
#endif
/***************************************************************
  Template instantiations :
   for num_dimensions=1,2,3
   T1,T2 : float, short, scaleT : Real
   T1,T2 : float, unsigned short, scaleT : Real

   T1,T2 : short, unsigned short, scaleT: Real (only for linking)
   T1=T2 : short, unsigned short, float, scaleT: Real
***************************************************************/

/*
 VC 5.0 has a bug that it cannot resolve the num_dimensions template-arg
 when using convert_array. You have to specify all template args explicitly.
 I do this here with macros to prevent other compilers breaking on it
 (notably VC 6.0...)
 */
#if defined(_MSC_VER) && (_MSC_VER < 1200)
#define INSTANTIATE(dim, type_in, type_out) \
   template \
   Array<dim,type_out> convert_array<dim,type_out, type_in>( \
                              Real& scale_factor, \
                              const Array<dim,type_in>& data_in, \
			      const NumericInfo<type_out> info2);
#else
#define INSTANTIATE(dim, type_in, type_out) \
   template \
   Array<dim,type_out> convert_array<>( \
			      Real& scale_factor, \
		              const Array<dim,type_in>& data_in, \
			      const NumericInfo<type_out> info2);
#endif



INSTANTIATE(1, float, short);
INSTANTIATE(1, float, unsigned short);
INSTANTIATE(1, short, float);
INSTANTIATE(1, unsigned short, float);
INSTANTIATE(1, unsigned short, short);
INSTANTIATE(1, short, unsigned short);

INSTANTIATE(1, short, short);
INSTANTIATE(1, unsigned short, unsigned short);
INSTANTIATE(1, float, float);

#if 0
template 
  void convert_array<2,float, short>(  Array<2,float>& data_out,
                              Real& scale_factor, 
                              const Array<2,short>& data_in, 
			      const NumericInfo<float> info2);
#endif

INSTANTIATE(2, float, short);
INSTANTIATE(2, float, unsigned short);
INSTANTIATE(2, short, float);
INSTANTIATE(2, unsigned short, float);
INSTANTIATE(2, unsigned short, short);
INSTANTIATE(2, short, unsigned short);

INSTANTIATE(2, short, short);
INSTANTIATE(2, unsigned short, unsigned short);
INSTANTIATE(2, float, float);

INSTANTIATE(3, float, short);
INSTANTIATE(3, float, unsigned short);
INSTANTIATE(3, short, float);
INSTANTIATE(3, unsigned short, float);
INSTANTIATE(3, unsigned short, short);
INSTANTIATE(3, short, unsigned short);

INSTANTIATE(3, short, short);
INSTANTIATE(3, unsigned short, unsigned short);
INSTANTIATE(3, float, float);


#undef INSTANTIATE

END_NAMESPACE_TOMO
