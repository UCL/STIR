
// $Id$
/*!
  \file 
  \ingroup Array 
  \brief implements the 1D specialisation of the Array class for broken compilers

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  $Date$
  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#if !defined(__stir_Array_H__) || !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) || !defined(T1) || !defined(T2)
#error This file should only be included in Array.cxx for half-broken compilers
#endif

/* Lines here should really be identical to what you find as 1D specialisation 
   in convert_array.cxx, except that  template statements are dropped.
   */
#ifndef T1equalT2
class convert_array_auxiliary<1, T1, T2, scaleT>
{
public:
  static inline void
    convert_array(Array<1,T2>& data_out, 
                  scaleT& scale_factor,
                  const Array<1,T1>& data_in)
  {
    
    const NumericInfo<T2> info2;
    
    find_scale_factor(scale_factor, data_in, info2);
    if (scale_factor == 0)
    {
      // data_in contains only 0
      data_out.fill(0);
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

#else // T1==T2
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
#endif

#undef T1
#undef T2
