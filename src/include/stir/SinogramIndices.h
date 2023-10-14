//
//
/*!
  \file
  \ingroup projdata

  \brief Definition of class stir::SinogramIndices

  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/



#ifndef __stir_SinogramIndices_h__
#define __stir_SinogramIndices_h__

#include "stir/SegmentIndices.h"

START_NAMESPACE_STIR

/*!
  \brief A very simple class to store all dincies to get a (2D) Sinogram
  \ingroup projdata 
*/
class SinogramIndices : public SegmentIndices
{
  typedef SegmentIndices base_type;
public:

  //! an empty constructor (sets everything to 0)
  inline  SinogramIndices();
  //! constructor taking view and segment number as arguments
  inline SinogramIndices( const int axial_pos_num,const int segment_num);

  //! get view number for const objects
  inline int axial_pos_num() const;

  //! get reference to view number
  inline int&  axial_pos_num();

 
  //! comparison operator, only useful for sorting
  /*! order : (0,1) < (0,-1) < (1,1) ...*/
  inline bool operator<(const SinogramIndices& other) const;

  //! test for equality
  inline bool operator==(const SinogramIndices& other) const;
  inline bool operator!=(const SinogramIndices& other) const;

private:
  int _axial_pos;

};

END_NAMESPACE_STIR

#include "stir/SinogramIndices.inl"

#endif
