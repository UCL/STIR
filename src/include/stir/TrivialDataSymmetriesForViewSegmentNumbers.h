//
// $Id$
//
/*
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
/*!

  \file
  \ingroup projdata

  \brief Declaration of class stir::TrivialDataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans

   $Date$

   $Revision$
*/
#ifndef __TrivialDataSymmetriesForViewSegmentNumbers_H__
#define __TrivialDataSymmetriesForViewSegmentNumbers_H__

#include "stir/DataSymmetriesForViewSegmentNumbers.h"

START_NAMESPACE_STIR

/*!
  \brief A class for encoding/finding NO symmetries. Works only on
  ViewSegmentNumbers (instead of Bin).

  This class is mainly useful if you need a DataSymmetriesForViewSegmentNumbers
  object (e.g. for RelatedViewgrams), but do not need/have projectors.

*/
class TrivialDataSymmetriesForViewSegmentNumbers : public DataSymmetriesForViewSegmentNumbers
{
public:

  virtual inline DataSymmetriesForViewSegmentNumbers * clone() const;


#if 0
  // TODO
  //! returns the range of the indices for basic view/segments
  virtual ViewSegmentIndexRange
    get_basic_view_segment_index_range() const;
#endif

  virtual inline void
    get_related_view_segment_numbers(vector<ViewSegmentNumbers>&, const ViewSegmentNumbers& v_s) const;

  virtual inline int
    num_related_view_segment_numbers(const ViewSegmentNumbers& v_s) const;

  /*! \brief given an arbitrary view/segment, find the basic view/segment
  
  in this class, \a v_s is unchanged, and the return value is always false.
  'v_s' is changed (i.e. it was NOT a basic view/segment).
  */  
  virtual inline bool
    find_basic_view_segment_numbers(ViewSegmentNumbers& v_s) const;

private:
  virtual bool blindly_equals(const root_type * const) const;

};

END_NAMESPACE_STIR

#include "stir/TrivialDataSymmetriesForViewSegmentNumbers.inl"

#endif

