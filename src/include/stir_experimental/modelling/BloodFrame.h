//
//
/*
    Copyright (C) 2005 - 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup modelling

  \brief Declaration of class stir::PlasmaData

  \author Charalampos Tsoumpas
 
*/

#ifndef __stir_modelling_BloodFrame_H__
#define __stir_modelling_BloodFrame_H__

#include "stir/common.h"
#include <vector>
#include <fstream>

START_NAMESPACE_STIR

class BloodFrame
{ 
public:
   //! default constructor
  inline BloodFrame();
 
  //! constructor of a frame, and its blood_counts_in_kBq, based on the acquired image. 
  inline BloodFrame(const unsigned int frame_num, const float blood_counts);

  //! constructor, mean time of a frame in seconds, and its blood_counts_in_kBq, based on the acquired image. 
  inline BloodFrame(const unsigned int frame_num, 
		    const float frame_start_time_in_s, 
		    const float frame_end_time_in_s, 
		    const float blood_counts);

  //! default destructor
  inline ~BloodFrame();
   
 //! set the time of the sample
  inline void set_frame_start_time_in_s( const float frame_start_time_in_s );
 //! set the time of the sample
  inline void set_frame_end_time_in_s( const float frame_end_time_in_s );
 //! get the time of the sample
  inline float get_frame_start_time_in_s() const; 
 //! get the time of the sample
  inline float get_frame_end_time_in_s() const; 
  //! set the frame number of the sample, if the plasma is based on the acquired image. 
  inline void set_frame_num( const unsigned int frame_num );
  //! get the frame number of the sample, if the plasma is based on the acquired image. 
  inline unsigned int get_frame_num() const;
 //! set the blood counts of the sample
  inline void set_blood_counts_in_kBq( const float blood_counts );
 //! get the blood counts of the sample
  inline float get_blood_counts_in_kBq() const; 
  
private : 
  float _blood_counts;
  float _frame_start_time_in_s;
  float _frame_end_time_in_s;
  unsigned int _frame_num;
};

END_NAMESPACE_STIR

#include "stir_experimental/modelling/BloodFrame.inl"

#endif //__stir_modelling_BloodFrame_H__
