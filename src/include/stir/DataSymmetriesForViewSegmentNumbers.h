//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata

  \brief Declaration of class stir::DataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans
  \author PARAPET project

*/
#ifndef __DataSymmetriesForViewSegmentNumbers_H__
#define __DataSymmetriesForViewSegmentNumbers_H__

#include "stir/ViewSegmentNumbers.h"
#include <vector>

START_NAMESPACE_STIR

#if 0
class ViewSegmentIndexRange;
#endif

/*!
  \ingroup projdata
  \brief A class for encoding/finding symmetries. Works only on
  ViewSegmentNumbers (instead of Bin).

  This class (mainly used in RelatedViewgrams and the projectors)
  is useful to store and use all information on symmetries
  common between the image representation and the projection data.

  The class mainly defines members to find \c basic ViewSegmentNumbers. These form a
  'basis' for all ViewSegmentNumbers in the sense that all ViewSegmentNumbers
  can be obtained by using symmetry operations on the 'basic' ones.

  \par Warning: This class wil be renamed/revised to work with \c ViewgramIndices instead.
*/
class DataSymmetriesForViewSegmentNumbers
{
public:
  virtual ~DataSymmetriesForViewSegmentNumbers();

  virtual DataSymmetriesForViewSegmentNumbers* clone() const = 0;

  //! Check equality
  /*! Implemented in terms of blindly_equals, after checking the type */
  bool operator==(const DataSymmetriesForViewSegmentNumbers&) const;

  //! Check inequality
  /*! Implemented in terms of operator==() */
  bool operator!=(const DataSymmetriesForViewSegmentNumbers&) const;

#if 0
  // TODO
  //! returns the range of the indices for basic view/segments
  virtual ViewSegmentIndexRange
    get_basic_view_segment_index_range() const = 0;
#endif

  //! fills in a vector with all the view/segments that are related to 'v_s' (including itself)
  virtual void get_related_view_segment_numbers(std::vector<ViewSegmentNumbers>&, const ViewSegmentNumbers& v_s) const = 0;

#if 0
  // not yet, as would need copying of vector
  //! fills in a vector with all the view/segments that are related to 'v_s' (including itself)
  virtual std::vector<ViewgramIndices>
    get_related_view_segment_numbers(const ViewgramIndices& ind) const
  {
  }
#endif

  //! returns the number of view_segment_numbers related to 'v_s'
  /*! The default implementation is in terms of get_related_view_segment_numbers, which will be
      slow of course */
  virtual int num_related_view_segment_numbers(const ViewSegmentNumbers& v_s) const;

  std::size_t num_related_viewgram_indices(const ViewgramIndices& ind) const
  {
    return static_cast<std::size_t>(num_related_view_segment_numbers(ind));
  }

  /*! \brief given an arbitrary view/segment, find the basic view/segment

  sets 'v_s' to the corresponding 'basic' view/segment and returns true if
  'v_s' is changed (i.e. it was NOT a basic view/segment).
  */
  virtual bool find_basic_view_segment_numbers(ViewSegmentNumbers& v_s) const = 0;

  /*! \brief test if a view/segment is 'basic'

  The default implementation uses find_basic_view_segment_numbers
  */
  virtual bool is_basic(const ViewSegmentNumbers& v_s) const;

protected:
  typedef DataSymmetriesForViewSegmentNumbers root_type;

  virtual bool blindly_equals(const root_type* const) const = 0;
};

END_NAMESPACE_STIR

#endif
