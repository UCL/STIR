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
*/
/*!
  \file
  \ingroup modelling
  \brief Implementations of inline functions of class stir::PlasmaData

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

  //! default constructor
PlasmaSample::PlasmaSample()
{ }
  //! constructor, time in s
PlasmaSample::
PlasmaSample(const double sample_time, const float plasma_sample_counts, const float blood_sample_counts)
{
  PlasmaSample::set_time_in_s( sample_time );
  PlasmaSample::set_blood_counts_in_kBq( blood_sample_counts );  
  PlasmaSample::set_plasma_counts_in_kBq( plasma_sample_counts );  
}

  //! default destructor
PlasmaSample::~PlasmaSample()
{ }
  
  //! set the time of the sample
void PlasmaSample::set_time_in_s( const double time )
{ PlasmaSample::_time=time ; }

  //! get the time of the sample
double PlasmaSample::get_time_in_s() const
{  return PlasmaSample::_time ; }

  //! set the blood counts of the sample 
void PlasmaSample::set_blood_counts_in_kBq( const float blood_counts )
{ PlasmaSample::_blood_counts=blood_counts ; }

  //! get the blood counts of the sample 
float PlasmaSample::get_blood_counts_in_kBq() const
{  return PlasmaSample::_blood_counts ; }

  //! get the plasma counts of the sample 
void PlasmaSample::set_plasma_counts_in_kBq( const float plasma_counts )
{ PlasmaSample::_plasma_counts=plasma_counts ; }

  //! get the plasma counts of the sample 
float PlasmaSample::get_plasma_counts_in_kBq() const
{  return PlasmaSample::_plasma_counts ; }



END_NAMESPACE_STIR
