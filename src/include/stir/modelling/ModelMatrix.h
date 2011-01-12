//
// $Id$
//
/*
    Copyright (C) 2006 - $Date$, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::ModelMatrix<num_param>
  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/

#ifndef __stir_modelling_ModelMatrix_H__
#define __stir_modelling_ModelMatrix_H__

#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
#include "stir/VectorWithOffset.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/Succeeded.h"
#include <fstream>
#include <iostream>

START_NAMESPACE_STIR

template <int num_param>
class ModelMatrix
{ 
public:
   inline ModelMatrix(); //!< default constructor

   inline ~ModelMatrix(); //!< default destructor

   /*! Implementation to read the model matrix from a text file
     \warning In this way the information about the calibration _is_uncalibrated and the counts _is_converted is not passed.
   */
   inline void read_from_file(const std::string input_string);

  //! Implementation to write the model matrix to a text file
   inline Succeeded write_to_file(const std::string output_string); 

   //! \name Functions to get parameters @{
   inline Array<2,float> get_model_array() const;
   inline const VectorWithOffset<float> get_model_array_sum() const;
   inline VectorWithOffset<float> get_time_vector() const;
  //!@}
   //! \name Functions to set parameters @{
   inline void set_model_array(const Array<2,float>& model_array);
   inline void set_time_vector(const VectorWithOffset<float>& time_vector);
  //! Function to set _is_calibrated boolean true or false
   inline void set_if_uncalibrated(const bool is_uncalibrated);
   inline void set_if_in_correct_scale(const bool in_correct_scale);
  //!@}

  //! Function to give the threshold_value to the all elements of the model_array which lower value than the threshold_value.  
   inline void threshold_model_array(const float threshold_value) ;

   /*! Function to divide with the calibration factor the model array. 
     Calibrated ModelMatrix means that the counts are in kBq/ml, while uncalibrated means that it will be to the same units as the reconstructed images.
    */
   inline void uncalibrate(const float cal_factor);

   /*! Function to multiply with the scale factor the model array. 
     Scaled ModelMatrix means that the counts are already scaled to the correct, while not scaled means that it needs to be scaled.
    */
   inline void scale_model_matrix(const float scale_factor);

   /*! Multiply with the duration to convert the count rate to total counts in the time frame.
     Converted ModelMatrix means that it is in total counts in respect to the time_frame_duration, 
     while not converted sets the _is_converted to false and means that it will be in "mean count rate".
    */
   inline void convert_to_total_frame_counts(const TimeFrameDefinitions& time_frame_definitions);

   /*! Multiplications of the model with the dynamic or the parametric images. 
     /todo Maybe it will be better to lie in a linear models class.
   */
   //@{
   //! multiply (transpose) model-matrix with dynamic image and add result to original \c parametric_image
   inline void 
    multiply_dynamic_image_with_model_and_add_to_input(ParametricVoxelsOnCartesianGrid & parametric_image,
                                                       const DynamicDiscretisedDensity & dynamic_image) const ;
   //! multiply (transpose) model-matrix with dynamic image (overwriting original content of \c parametric_image)
   /*! \todo current implementation first fills first argument with 0 and then calls 
    multiply_dynamic_image_with_model_and_add_to_input(). This is somewhat inefficient.
   */
   inline void 
    multiply_dynamic_image_with_model(ParametricVoxelsOnCartesianGrid & parametric_image,
                                      const DynamicDiscretisedDensity & dynamic_image) const ;
   //! multiply model-matrix with parametric image and add result to original \c dynamic_image
   inline void
     multiply_parametric_image_with_model_and_add_to_input(DynamicDiscretisedDensity & dynamic_image,
                                                          const ParametricVoxelsOnCartesianGrid & parametric_image ) const ; 
   //! multiply model-matrix with parametric image (overwriting original content of \c dynamic_image)
   /*! \todo current implementation first fills first argument with 0 and then calls 
    multiply_dynamic_image_with_model_and_add_to_input(). This is somewhat inefficient.
   */
   inline void
    multiply_parametric_image_with_model(DynamicDiscretisedDensity & dynamic_image,
                                         const ParametricVoxelsOnCartesianGrid & parametric_image ) const ; 

   inline void 
     normalise_parametric_image_with_model_sum( ParametricVoxelsOnCartesianGrid & parametric_image_out,
                                           const ParametricVoxelsOnCartesianGrid & parametric_image ) const ; 
  //@}
private:

   //! At the moment it has the form of _model_array[param_num][frame_num].
   Array<2,float> _model_array;
   VectorWithOffset<float> _time_vector;
   bool _is_uncalibrated ;   
   bool _in_correct_scale ;
   bool _is_converted_to_total_counts;
};

END_NAMESPACE_STIR

#include "stir/modelling/ModelMatrix.inl"

#endif //__stir_modelling_ModelMatrix_H__
