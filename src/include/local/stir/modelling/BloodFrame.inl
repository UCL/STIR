//
// $Id$
//
/*
    Copyright (C) 2005 - $Date$, Hammersmith Imanet Ltd
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

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::PlasmaData

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

  //! default constructor
BloodFrame::BloodFrame()
{ }
  //! constructor, time in s
BloodFrame::BloodFrame(const unsigned int frame_num, const float sample_time, const float blood_sample_counts)
{
  BloodFrame::set_frame_num( frame_num );
  BloodFrame::set_time_in_s( sample_time );
  BloodFrame::set_blood_counts_in_kBq( blood_sample_counts );  
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
  
  //! set the time of the sample
void BloodFrame::set_time_in_s( const float time )
{ BloodFrame::_time=time ; }

  //! get the time of the sample
float BloodFrame::get_time_in_s() const
{  return BloodFrame::_time ; }

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
