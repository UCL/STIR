//
// $Id$
//
/*!

  \file
  \ingroup projdata

  \brief Declaration of class TrivialDataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans

   $Date$

   $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
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

};

END_NAMESPACE_STIR

#include "stir/TrivialDataSymmetriesForViewSegmentNumbers.inl"

#endif

