//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#ifndef __IndexRange_H__
#define __IndexRange_H__

/*!
  \file

  \brief This file defines the stir::IndexRange class.

  \author Kris Thielemans
  \author PARAPET project



*/
#include "stir/VectorWithOffset.h"
#include "stir/BasicCoordinate.h"
#include "stir/IndexRangeFwd.h"

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
template <int num_dimensions, typename indexT>
class IndexRange : public VectorWithOffset<IndexRange<num_dimensions - 1, indexT>, indexT>
{
protected:
  typedef VectorWithOffset<IndexRange<num_dimensions - 1, indexT>, indexT> base_type;

public:
  //! typedefs such that we do not need to have \a typename wherever we use iterators
  typedef typename base_type::iterator iterator;
  typedef typename base_type::const_iterator const_iterator;

  //! Empty range
  inline IndexRange();

#ifndef SWIG
  // SWIG bug prevents using base_type here. Leads to problems with num_dimensions-1
  //! Make an IndexRange from the base type
  inline IndexRange(const base_type& range);
#endif

  //! Copy constructor
  inline IndexRange(const IndexRange<num_dimensions, indexT>& range);

  //! Construct a regular range given by all minimum indices and all maximum indices.
  inline IndexRange(const BasicCoordinate<num_dimensions, indexT>& min, const BasicCoordinate<num_dimensions, indexT>& max);

  //! Construct a regular range given by sizes (minimum indices will be 0)
  inline IndexRange(const BasicCoordinate<num_dimensions, indexT>& sizes);

  //! return the total number of elements in this range
  inline size_t size_all() const;

  // these are derived from VectorWithOffset
  //  TODO these should be overloaded, to set regular_range as well.
  /*
  const IndexRange<num_dimensions-1>& operator[](indexT i) const
  { return range[i]; }

  IndexRange<num_dimensions-1>& operator[](indexT i)
  { return range[i]; }
  */

  //! comparison operator
  inline bool operator==(const IndexRange<num_dimensions, indexT>&) const;
  inline bool operator!=(const IndexRange<num_dimensions, indexT>&) const;

  //! checks if the range is 'regular'
  inline bool is_regular() const;

  //! find regular range, returns false if the range is not regular
  inline bool get_regular_range(BasicCoordinate<num_dimensions, indexT>& min, BasicCoordinate<num_dimensions, indexT>& max) const;

private:
  //! enum to encode the current knowledge about regularity
  enum is_regular_type
  {
    regular_true,
    regular_false,
    regular_to_do
  };

  //! variable storing the current knowledge about regularity
  mutable is_regular_type is_regular_range;
};

//! The (simple) 1 dimensional specialisation of IndexRange
template <class indexT>
class IndexRange<1, indexT>
{
public:
  typedef size_t size_type;
  inline IndexRange();
  inline IndexRange(const indexT min, const indexT max);

  inline IndexRange(const BasicCoordinate<1, indexT>& min, const BasicCoordinate<1, indexT>& max);

  inline IndexRange(const size_type length);
  inline IndexRange(const BasicCoordinate<1, indexT>& size);

  inline indexT get_min_index() const;
  inline indexT get_max_index() const;
  inline size_type get_length() const;
  //! return the total number of elements in this range
  inline size_t size_all() const;

  inline bool operator==(const IndexRange<1, indexT>& range2) const;

  //! checks if the range is 'regular' (always true for the 1d case)
  inline bool is_regular() const;

  //! fills in min and max, and returns true
  inline bool get_regular_range(BasicCoordinate<1, indexT>& min, BasicCoordinate<1, indexT>& max) const;
  //! resets to new index range
  inline void resize(const indexT min_index, const indexT max_index);

private:
  indexT min;
  indexT max;
};

END_NAMESPACE_STIR

#include "stir/IndexRange.inl"

#endif
