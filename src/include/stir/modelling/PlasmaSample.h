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
  \brief Declaration of class stir::PlasmaData
  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/

#ifndef __stir_modelling_PlasmaSample_H__
#define __stir_modelling_PlasmaSample_H__

#include "stir/common.h"
#include <vector>
#include <fstream>

START_NAMESPACE_STIR

class PlasmaSample
{ 
public:
   //! default constructor
  inline PlasmaSample();

  /*!  A constructor : constructs a PlasmaSample object \n
    \param sample_time is the time in \a seconds relativily to the start of the scan.
    \param plasma_sample_counts is the activity of plasma at the sample_time (assumed to be in \a kBq/ml)
    \param blood_sample_counts is the activity of blood at the sample_time (assumed to be in \a kBq/ml)    
  */ 
  inline PlasmaSample( const float sample_time, const float plasma_sample_counts, const float blood_sample_counts);

  //! default destructor
  inline ~PlasmaSample();

  //! \name Functions to get parameters @{
 //! get the time of the sample
  inline float get_time_in_s() const; 
 //! get the blood counts of the sample
  inline float get_blood_counts_in_kBq() const; 
 //! get the plasma counts of the sample @}
  inline float get_plasma_counts_in_kBq() const; 

  //! \name Functions to set parameters @{
  //! set the time of the sample
  inline void set_time_in_s( const float time );
  //! set the blood counts of the sample
  inline void set_blood_counts_in_kBq( const float blood_counts );
  //! set the plasma counts of the sample @}
  inline void set_plasma_counts_in_kBq( const float plasma_counts ); 



  
private : 
  float _blood_counts;
  float _plasma_counts;
  float _time;
};

END_NAMESPACE_STIR

#include "stir/modelling/PlasmaSample.inl"

#endif //__stir_modelling_PlasmaSample_H__
