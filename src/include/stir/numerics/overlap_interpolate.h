//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup numerics

  \brief Declaration of stir::overlap_interpolate

  \author Kris Thielemans
  \author PARAPET project

*/

#ifndef __stir_numerics_overlap_interpolate__H__
#define  __stir_numerics_overlap_interpolate__H__

#include "stir/common.h"

START_NAMESPACE_STIR

template <class T> class VectorWithOffset;

/*!
  \ingroup numerics
 
  \brief 'overlap' interpolation (i.e. count preserving) for vectors.
  \see  overlap_interpolate(const out_iter_t out_begin, const out_iter_t out_end, 
		     const out_coord_iter_t out_coord_begin, const out_coord_iter_t out_coord_end,
		     const in_iter_t in_begin, in_iter_t in_end,
		     const in_coord_iter_t in_coord_begin, const in_coord_iter_t in_coord_end,
		     const bool only_add_to_output=false, const bool assign_rest_with_zeroes)
*/
template <typename T>
void
overlap_interpolate(VectorWithOffset<T>& out_data, 
		    const VectorWithOffset<T>& in_data,
		    const float zoom, 
		    const float offset, 
		    const bool assign_rest_with_zeroes = true);


/*! \ingroup numerics
  \brief 'overlap' interpolation for iterators, with arbitrary 'bin' sizes.

  This type of interpolation considers the data as the samples of
  a step-wise function. The interpolated array again represents a
  step-wise function, such that the counts (i.e. integrals) are
  preserved.

  In and out data are specified using iterators. For each, there is 
  also a pair of iterators specifying the coordinates of the edges
  of the 'bins' (or 'boxes') in some arbitrary coordinate system 
  (common between in and out parameters of course). Note that there
  should be one more coordinate than data (i.e. you have to specify
  the last edge as well). This is (only) checked with assert() statements.
  
  \param only_add_to_output
  If \c false will overwrite any data present in the output (aside from possibly
  the tails: see \c assign_rest_with_zeroes).
  If \c true, results will be added to the data.
  \param assign_rest_with_zeroes
  If \c false does not set values in the \c out range which do not overlap with
  \c in range.
  If \c true those data are set to 0. 

  \warning when the out iterators point to an integral type, there is no 
  rounding but truncation.

  \par Examples:
  Given 2 arrays and zoom and offset parameters
  \code
    Array<1,float> in = ...;
    Array<1,float> out = ...;
    float zoom = ...; float offset = ...;
  \endcode
  the following pieces of code should give the same result (this is tested
  in test_interpolate):
  \code
    overlap_interpolate(out, in, zoom, offset, true);
  \endcode
  and
  \code
    Array<1,float> in_coords(in.get_min_index(), in.get_max_index()+1);
    for (int i=in_coords.get_min_index(); i<=in_coords.get_max_index(); ++i)
      in_coords[i]=i-.5F;
    Array<1,float> out_coords(out.get_min_index(), out.get_max_index()+1);
    for (int i=out_coords.get_min_index(); i<=out_coords.get_max_index(); ++i)
      out_coords[i]=(i-.5F)/zoom+offset;
    overlap_interpolate(out.begin(), out.end(),
			out_coords.begin(), out_coords.end(),
			in.begin(), in.end(),
			in_coords.begin(), in_coords.end()
			);
  \endcode
    
  \par Implementation details:

  Because this implementation works for arbitrary (numeric) types, it
  is slightly more complicated than would be necessary for (say) floats.
  In particular,
  <ul>
  <li> we do our best to avoid creating temporary objects</li>
  <li> we zero values by using multiplication with 0. This is to allow 
  the case where assignment with an int (or float) does not exist
  (in particular, in our higher dimensional arrays).</li>
  </ul>
  The implementation is inline to avoid problems with template
  instantiations.
  \par History:
  <ul>
  <li> first version by Kris Thielemans</li>
  </ul>
 */
template <typename out_iter_t, typename out_coord_iter_t,
	  typename in_iter_t, typename in_coord_iter_t>
inline
void
 overlap_interpolate(const out_iter_t out_begin, const out_iter_t out_end, 
		     const out_coord_iter_t out_coord_begin, const out_coord_iter_t out_coord_end,
		     const in_iter_t in_begin, in_iter_t in_end,
		     const in_coord_iter_t in_coord_begin, const in_coord_iter_t in_coord_end,
		     const bool only_add_to_output=false, const bool assign_rest_with_zeroes=true);

END_NAMESPACE_STIR

#include "stir/numerics/overlap_interpolate.inl"

#endif
