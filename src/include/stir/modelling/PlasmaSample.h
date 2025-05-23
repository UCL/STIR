//
//
/*
    Copyright (C) 2005 - 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_modelling_PlasmaSample_H__
#define __stir_modelling_PlasmaSample_H__

#include "stir/common.h"
#include <vector>
#include <fstream>

START_NAMESPACE_STIR

/*!
 \ingroup modelling

 A class for storing radiotracer concentration in a blood sample. Concentrations in plasma and the (overall) blood-concentration
 are stored.

 \todo This currently assumes sampling, while in practice we often have data accumulated over time, but this can
 currently not be encoded.
*/
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
  inline PlasmaSample( const double sample_time, const float plasma_sample_counts, const float blood_sample_counts);

  //! default destructor
  inline ~PlasmaSample();

  //! \name Functions to get parameters 
  //@{
  //! get the time of the sample
  inline double get_time_in_s() const; 
  //! get the blood counts of the sample
  inline float get_blood_counts_in_kBq() const; 
  //! get the plasma counts of the sample 
  inline float get_plasma_counts_in_kBq() const; 
  //@}

  //! \name Functions to set parameters 
  //@{
  //! set the time of the sample
  inline void set_time_in_s( const double time );
  //! set the blood counts of the sample
  inline void set_blood_counts_in_kBq( const float blood_counts );
  //! set the plasma counts of the sample
  inline void set_plasma_counts_in_kBq( const float plasma_counts ); 
  //@}


  
private : 
  float _blood_counts;
  float _plasma_counts;
  double _time;
};

END_NAMESPACE_STIR

#include "stir/modelling/PlasmaSample.inl"

#endif //__stir_modelling_PlasmaSample_H__
