//
//
/*!
  \file
  \ingroup projdata

  \brief Definition of class stir::SegmentIndices

  \author Kris Thielemans

*/
/*
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SegmentIndices_h__
#define __stir_SegmentIndices_h__

#include "stir/common.h"

START_NAMESPACE_STIR

/*!
  \brief A very simple class to store segment numbers and any other
  indices that define a segment
  \ingroup projdata
*/
class SegmentIndices
{
public:
  //! constructor segment number as arguments
  explicit inline SegmentIndices(const int segment_num = 0);

  //! get segment number for const objects
  inline int segment_num() const;

  //! get reference to segment number
  inline int& segment_num();

  //! comparison operator, only useful for sorting
  /*! In future, there will be multiple indices, and order will then be based as in
      <code>(0,1) < (0,-1) < (1,1) ...</code>
  */
  inline bool operator<(const SegmentIndices& other) const;

  //! test for equality
  inline bool operator==(const SegmentIndices& other) const;
  inline bool operator!=(const SegmentIndices& other) const;

private:
  int _segment;
};

END_NAMESPACE_STIR

#include "stir/SegmentIndices.inl"

#endif
