//
//
/*
    Copyright (C) 2005 - 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::PlasmaData

  \author Charalampos Tsoumpas

*/

START_NAMESPACE_STIR

  //! default constructor
BloodFrame::BloodFrame()
{ }
  //! constructor, time in s
BloodFrame::BloodFrame(const unsigned int frame_num, const float frame_start_time_in_s, const float frame_end_time_in_s, const float blood_counts)
{
  BloodFrame::set_frame_num( frame_num );
  BloodFrame::set_frame_start_time_in_s( frame_start_time_in_s );
  BloodFrame::set_frame_end_time_in_s( frame_end_time_in_s );
  BloodFrame::set_blood_counts_in_kBq( blood_counts );  
}

 //! constructor, frame number, if the plasma is based on the acquired image. 
BloodFrame::BloodFrame(const unsigned int frame_num, const float blood_sample_counts)
{
  BloodFrame::set_frame_num( frame_num );
  BloodFrame::set_blood_counts_in_kBq( blood_sample_counts );  
}

  //! default destructor
BloodFrame::~BloodFrame()
{ }
  
  //! set the start_time of the sample
void BloodFrame::set_frame_start_time_in_s( const float frame_start_time_in_s )
{ BloodFrame::_frame_start_time_in_s=frame_start_time_in_s ; }

  //! get the start_time of the sample
float BloodFrame::get_frame_start_time_in_s() const
{  return BloodFrame::_frame_start_time_in_s ; }

  //! set the start_time of the sample
void BloodFrame::set_frame_end_time_in_s( const float frame_end_time_in_s )
{ BloodFrame::_frame_end_time_in_s=frame_end_time_in_s ; }

  //! get the start_time of the sample
float BloodFrame::get_frame_end_time_in_s() const
{  return BloodFrame::_frame_end_time_in_s ; }

  //! set the frame number of the sample, if the plasma is based on the acquired image. 
void BloodFrame::set_frame_num( const unsigned int frame_num )
{ BloodFrame::_frame_num=frame_num ; }

  //! get the frame number of the sample, if the plasma is based on the acquired image. 
unsigned int  BloodFrame::get_frame_num() const
{  return BloodFrame::_frame_num ; }

  //! set the blood counts of the sample 
void BloodFrame::set_blood_counts_in_kBq( const float blood_counts )
{ BloodFrame::_blood_counts=blood_counts ; }

  //! get the blood counts of the sample 
float BloodFrame::get_blood_counts_in_kBq() const
{  return BloodFrame::_blood_counts ; }



END_NAMESPACE_STIR
