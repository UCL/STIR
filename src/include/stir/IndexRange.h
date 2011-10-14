//
// $Id$
//
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

#ifndef __IndexRange_H__
#define __IndexRange_H__

/*!
  \file 
 
  \brief This file defines the stir::IndexRange class.

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$

*/
#include "stir/VectorWithOffset.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

/*!
  \ingroup Array
  \brief  This class defines ranges which can be 'irregular'.

  This class allows construction and basic manipulation of 'irregular'
  (but not completely arbitrary) ranges. As the class diagram shows,
  an IndexRange<n> is basically a VectorWithOffset with elements
  of type IndexRange<n-1>. This recursion ends in IndexRange<1>
  which is simply a pair of numbers, given the start and end of the
  1D range.
  
  This means that the outer index runs over an interval of integers.
  The next level of indices again runs over such an interval, but
  which interval can depend on the value of the outer index.

  For instance for a 2D range of indices \a i, \a j, the outer index
  \a i could run from 1 to 2, and for \a i=1, \a j could run from 4 to 6,
  while for \a i=2, \a j could run from 6 to 8.

  Facilities are provided for constructing 'regular' ranges (where the
  range of the inner indices does not depend on the value of the 
  outer indices). However, storage is currently not optimised for the
  regular case.

  Example of usage:
  \code
  IndexRange<3> range = construct_me_an_index_range();
  int outer_index = range.get_min_index();
  while(outer_index <= range.get_max_index())
  {
    int level_2_min_index = range[outer_index].get_min_index();
    ...
  }
  \endcode
*/
template <int num_dimensions>
class IndexRange : public VectorWithOffset<IndexRange<num_dimensions-1> >
{
protected:
  typedef VectorWithOffset<IndexRange<num_dimensions-1> > base_type;

public:
  //! typedefs such that we do not need to have \a typename wherever we use iterators 
  typedef typename base_type::iterator iterator;
  typedef typename base_type::const_iterator const_iterator;

  //! Empty range
  inline IndexRange();
  
  //! Make an IndexRange from the base type
  inline IndexRange(const VectorWithOffset<IndexRange<num_dimensions-1> >& range);

  //! Copy constructor
  inline IndexRange(const IndexRange<num_dimensions>& range);

  //! Construct a regular range given by all minimum indices and all maximum indices.
  inline IndexRange(
		    const BasicCoordinate<num_dimensions, int>& min,
		    const BasicCoordinate<num_dimensions, int>& max);

  //! Construct a regular range given by sizes (minimum indices will be 0)
  inline IndexRange(const BasicCoordinate<num_dimensions, int>& sizes);
  
  //these are derived from VectorWithOffset
  // TODO these should be overloaded, to set regular_range as well.
  /*
  const IndexRange<num_dimensions-1>& operator[](int i) const
  { return range[i]; }

  IndexRange<num_dimensions-1>& operator[](int i)
  { return range[i]; }
  */

  //! comparison operator
  inline bool operator==(const IndexRange<num_dimensions>&) const;
  inline bool operator!=(const IndexRange<num_dimensions>&) const;

  //! checks if the range is 'regular'
  inline bool is_regular() const;

  //! find regular range, returns false if the range is not regular
  bool get_regular_range(
			 BasicCoordinate<num_dimensions, int>& min,
			 BasicCoordinate<num_dimensions, int>& max) const;

#ifdef STIR_NO_MUTABLE
  //! checks if the range is 'regular'
  inline bool is_regular();

  //! find regular range
  bool get_regular_range(
			 BasicCoordinate<num_dimensions, int>& min,
			 BasicCoordinate<num_dimensions, int>& max);
#endif

private:
  //! enum to encode the current knowledge about regularity
  enum is_regular_type {regular_true, regular_false, regular_to_do};

  //! variable storing the current knowledge about regularity
#ifndef STIR_NO_MUTABLE
  mutable 
#endif
    is_regular_type is_regular_range;
};


//! The (simple) 1 dimensional specialisation of IndexRange     
template<>
class IndexRange<1>
{
public:
  inline IndexRange();
  inline IndexRange(const int min, const int max);

  inline IndexRange(const BasicCoordinate<1,int>& min, 
		    const BasicCoordinate<1,int>& max);

  inline IndexRange(const int length);
  inline IndexRange(const BasicCoordinate<1, int>& size);

  inline int get_min_index() const;
  inline int get_max_index() const;
  inline int get_length() const;

  inline bool operator==(const IndexRange<1>& range2) const;

  //! checks if the range is 'regular' (always true for the 1d case)
  inline bool is_regular() const;

  //! fills in min and max, and returns true
  inline bool get_regular_range(
				BasicCoordinate<1, int>& min,
				BasicCoordinate<1, int>& max) const;
  //! resets to new index range
  inline void resize(const int min_index, const int max_index);

private:
  int min; 
  int max;
};


END_NAMESPACE_STIR

#include "stir/IndexRange.inl"

#endif


