//
//

/*!
  \file
  \ingroup projdata

  \brief Declaration of class stir::Bin
  
  \author Nikos Efthimiou
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2016, University of Hull
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
#ifndef __stir_Bin_H__
#define __stir_Bin_H__


#include "stir/common.h"


START_NAMESPACE_STIR
/*!   \ingroup projdata
 \brief
 A class for storing coordinates and value of a single projection bin.

 The timing position reflect the detection time difference between the two events.
 It is a multiple of the delta t of the least significant clock bit.

 \details In order to go around the timing pos which is set only in TOF reconstruction
 we set it, by default, to zero. If it is zero in both bins then it is omitted from the
 comparison.
 \warning Comparison between bin with and without timing position is of high risk!

 \warning Constructors with default values were removed.

 \warning Temporarily the timing_pos_num is not taken into account when comparing two bins.
*/

class Bin
{
public: 
  //! default constructor
  inline Bin();

  //!  A constructor : constructs a bin with value (defaulting to 0)
  inline Bin(int segment_num,int view_num, int axial_pos_num,
    int tangential_pos_num,float bin_value);

  inline Bin(int segment_num, int view_num, int axial_pos_num,
             int tangential_pos_num);

  inline Bin(int segment_num, int view_num, int axial_pos_num,
             int tangential_pos_num, int timing_pos_num, float bin_value);

  inline Bin(int segment_num, int view_num, int axial_pos_num,
             int tangential_pos_num, int timing_pos_num);
  
  //!get axial position number
  inline int axial_pos_num()const;
  //! get segmnet number
  inline int segment_num()const; 
  //! get tangential position number
  inline int tangential_pos_num()  const; 
  //! get view number
  inline int view_num() const; 
  //! get timing position number
  inline int timing_pos_num() const;
  
  inline int& axial_pos_num(); 
  inline int& segment_num(); 
  inline int& tangential_pos_num(); 
  inline int& view_num(); 
  
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
  int  timing_pos;
  float bin_value;
  
  
};



END_NAMESPACE_STIR


#include "stir/Bin.inl"

#endif //__Bin_H__
