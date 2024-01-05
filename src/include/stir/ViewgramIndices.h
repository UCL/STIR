//
//
/*!
  \file
  \ingroup projdata

  \brief Definition of class stir::ViewgramIndices

  \author Kris Thielemans

*/
/*
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ViewgramIndices_h__
#define __stir_ViewgramIndices_h__

#include "stir/SegmentIndices.h"

START_NAMESPACE_STIR

/*!
  \brief A very simple class to store all dincies to get a (2D) Viewgram
  \ingroup projdata
*/
class ViewgramIndices : public SegmentIndices
{
  typedef SegmentIndices base_type;

public:
  //! an empty constructor (sets everything to 0)
  //TODOTOF remove default
  inline ViewgramIndices();
  //! constructor taking view and segment number as arguments
  inline ViewgramIndices(const int view_num, const int segment_num, const int timing_pos_num=0);

  //! get view number for const objects
  inline int view_num() const;

  //! get reference to view number
  inline int& view_num();

  //! comparison operator, only useful for sorting
  /*! order : (0,1) < (0,-1) < (1,1) ...*/
  inline bool operator<(const ViewgramIndices& other) const;

  //! test for equality
  inline bool operator==(const ViewgramIndices& other) const;
  inline bool operator!=(const ViewgramIndices& other) const;

private:
  int _view;
};

END_NAMESPACE_STIR

#include "stir/ViewgramIndices.inl"

#endif
