//
//

/*!
  \file
  \ingroup projdata

  \brief Declaration of class stir::Bin
  

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_Bin_H__
#define __stir_Bin_H__


#include "stir/common.h"


START_NAMESPACE_STIR
/*!   \ingroup projdata
 \brief
 A class for storing coordinates and value of a single projection bin.

 The \c time_frame member defaults to 1 and needs to be set explicitly, e.g. when
 handling list mode data.
*/

class Bin
{
public: 
  //! default constructor
  inline Bin();

  //!  A constructor : constructs a bin with value (defaulting to 0)
  inline Bin(int segment_num,int view_num, int axial_pos_num,
    int tangential_pos_num,float bin_value=0);
  
  //!get axial position number
  inline int axial_pos_num()const;
  //! get segmnet number
  inline int segment_num()const; 
  //! get tangential position number
  inline int tangential_pos_num()  const; 
  //! get view number
  inline int view_num() const; 
  //! get time-frame number (1-based)
  inline int time_frame_num() const;
  
  inline int& axial_pos_num(); 
  inline int& segment_num(); 
  inline int& tangential_pos_num(); 
  inline int& view_num(); 
  inline int& time_frame_num();
  
  
  //! get an empty copy
  inline Bin get_empty_copy() const;
  
  //! get the value after forward projection 
  inline float get_bin_value()const; 
  //! set the value to be back projected 
  inline void set_bin_value( float v );
  
  //! accumulate voxel's contribution during forward projection 
  inline Bin&  operator+=(const float dx);
  //! multiply bin values
  inline Bin& operator*=(const float dx);
  //! divide bin values
  //! \todo It is zero division proof in a similar way to divide<,,>(), though I am
  //! not sure if it should be.
  inline Bin& operator/=(const float dx);
  
  //! comparison operators
  inline bool operator==(const Bin&) const;
  inline bool operator!=(const Bin&) const;
  
private :
  // shared_ptr<ProjDataInfo> proj_data_info_ptr; 
  
  int  segment;
  int  view; 
  int  axial_pos; 
  int  tangential_pos; 
  float bin_value;
  int time_frame;
  
  
};

//! A small bundle of a Bin and its additive correction value
//!
struct BinAndCorr
{
    Bin my_bin;
    float my_corr;
};



END_NAMESPACE_STIR


#include "stir/Bin.inl"

#endif //__Bin_H__
