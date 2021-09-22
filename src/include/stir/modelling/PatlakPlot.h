//
//
/*
    Copyright (C) 2006 - 2011, Hammersmith Imanet Ltd
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup modelling
  \brief Implementation of functions of class stir::PatlakPlot

  \author Charalampos Tsoumpas
  \author Kris Thielemans
*/


#ifndef __stir_modelling_PatlakPlot_H__
#define __stir_modelling_PatlakPlot_H__

#include "stir/modelling/KineticModel.h"
#include "stir/modelling/ModelMatrix.h"
#include "stir/modelling/PlasmaData.h"
#include "stir/Succeeded.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

//!
/*!
  \ingroup modelling
  \brief Patlak kinetic model

  Model suitable for irreversible tracers such as FDG and FLT. See
 
  - Patlak C S, Blasberg R G, Fenstermacher J D (1985)  
      <i>Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data,</i> {J Cereb Blood Flow Metab 3(1): p. 1-7.

  - Patlak C S, Blasberg R G (1985)
    <i>Experimental and Graphical evaluation of blood-to-brain transfer constant from multiple-time uptake data: Generalizations,</i>
    J Cereb Blood Flow Metab 5: p. 584-90. 

  The kinetic model is
  \f[
     C(t) =Ki*\int{C_p(t)\,dt}+V C_p(t)
  \f]
  with \f$C(t)\f$ the tissue TAC and \f$C_p(t)\f$ the (plasma) input function, with \f$Ki\f$ the slope
  and \f$V\f$ the intercept. \f$t\f$ is a time-frame index (i.e. the activity/input function is
  integrated over the time frame).

  \par Example .par file
  \verbatim
  Patlak Plot Parameters:=

  time frame definition filename := frames.txt
  starting frame := 23
  calibration factor := 9000
  blood data filename :=  blood_file.txt
  ; In seconds
  Time Shift := 0
  ; work-around current lack of STIR unit info
  In total counts := 0 ; defaults to 0, set to 1 for images in activity
  ; enable this for fully calibrated images from your scanner
  In correct scale := 0 ; defaults to 0

  ; enable this if you want to use weighted linear regression, assuming
  ; that each voxel and time frame is Poisson distributed
  Poisson distributed images := 0 ; defaults to 0

  end Patlak Plot Parameters:=
  \endverbatim

  \warning
  - The dynamic images will be calibrated only if the calibration factor is given. 
  - The \c if_total_cnt is set to \c true the Dynamic Image will be assumed to have the total number of
    counts while if set to \c false it will have the \c total_number_of_counts/get_duration(frame_num).
  - The dynamic images will always be assumed to be without decay correction.
  - The plasma data is assumed to be without decay correction.

  This class provides functionality for iterative estimation as well as "ordinary"
  linear regression (see \c apply_linear_regression()).

  \todo Should be derived from LinearModels, but when non-linear models will be introduced, as well.  
*/
class PatlakPlot : public RegisteredParsingObject<PatlakPlot, KineticModel> 
{
  public:
  //! Name which will be used when parsing a PatlakPlot object
  static const char * const registered_name; 

   PatlakPlot();   //!< Default constructor (calls set_defaults())
   ~PatlakPlot();   //!< default destructor
   /*! \name Functions to get parameters */
   //@{
    //! Simply gets model matrix, if it has been already stored.
    ModelMatrix<2> get_model_matrix() const;
    //! Creates model matrix from plasma data (Must be already sorted in appropriate frames).
    ModelMatrix<2> get_model_matrix(const PlasmaData& plasma_data, 
				    const TimeFrameDefinitions& time_frame_definitions, 
				    const unsigned int starting_frame);
    //! Returns the frame that the PatlakPlot linearization is assumed to be valid.
    unsigned int
      get_starting_frame() const ;
    //! Returns the number of the last frame available. 
    unsigned int
      get_ending_frame() const ;
    //! Returns the TimeFrameDefinitions that the PatlakPlot linearization is assumed to be valid: ChT::Check
    TimeFrameDefinitions 
      get_time_frame_definitions() const ;
    //!@}
    /*! \name Functions to set parameters*/
    //@{  
    void set_model_matrix(ModelMatrix<2> model_matrix) ;     //!< Simply set model matrix 
    //@}

    //! Multiplies the dynamic image with the model gradient. 
    /*!  For a linear model the model gradient is the transpose of the model matrix. 
      So, the dynamic image is "projected" from time domain to the parameter domain.

      \todo Should be a virtual function declared in the KineticModel class.
    */  
    virtual void
      multiply_dynamic_image_with_model_gradient(ParametricVoxelsOnCartesianGrid & parametric_image,
						 const DynamicDiscretisedDensity & dyn_image) const;
    //! Multiplies the dynamic image with the model gradient and add to original \c parametric_image 
    /*! \todo Should be a virtual function declared in the KineticModel class.
    */
    virtual void
      multiply_dynamic_image_with_model_gradient_and_add_to_input(ParametricVoxelsOnCartesianGrid & parametric_image,
						 const DynamicDiscretisedDensity & dyn_image) const;

    //! Multiplies the parametric image with the model matrix to get the corresponding dynamic image.
    /*! \todo Should be a virtual function declared in the KineticModel class.
    */
    virtual void
      get_dynamic_image_from_parametric_image(DynamicDiscretisedDensity & dyn_image,
					      const ParametricVoxelsOnCartesianGrid & par_image) const;

    //! estimate parametric images from dynamic images
    /*! This performs Patlak Linear regression is applied to the data by minimising
       \f[\sum_t w_t ( C(t)/C_p(t) - (Ki*\int{C_p(t)\,dt}/C_p(t)+V))^2 \f]
       Weights can currently be chosen as either 1, or by assuming that \f$C(t)\f$
       is Poisson distributed (potentially after converting to "counts"),
       which is a reasonable approximation for OSEM images (although somewhat dangerous
       for noisy data). The latter is chosen if
       \c _assume_poisson_distribution is \c true.

       \todo There is currently no check if the time frame definitions from \a dyn_image are
      the same as the ones encoded in the model.
    */
    void 
      apply_linear_regression(ParametricVoxelsOnCartesianGrid & par_image, const DynamicDiscretisedDensity & dyn_image) const;

    void set_defaults();

    Succeeded set_up(); 

  bool _if_cardiac;   //!< Switches between cardiac and brain data
  unsigned int _starting_frame;   //!< Starting frame to apply the model
  float _cal_factor;   //!< Calibration Factor, maybe to be removed.
  float _time_shift;   //!< Shifts the time to fit the timing of Plasma Data with the Projection Data.
  bool _in_correct_scale; //!< Switch to scale or not the model_matrix to the correct scale, according to the appropriate scale factor.
  bool _in_total_cnt;   //!< Switch to choose the values of the model to be in total counts or in mean counts.
  bool _assume_poisson_distribution; //!< Assume that image is Poisson distributed, and weight linear regression accordingly (see \c apply_linear_regression())
  std::string _blood_data_filename;   //!< Name of file in which the input function is stored
  PlasmaData _plasma_frame_data;    //!< Stores the plasma data into frames for brain studies
  std::string _time_frame_definition_filename;   //!< name of file to get frame definitions
  TimeFrameDefinitions _frame_defs;   //!< TimeFrameDefinitions

 private:
  void create_model_matrix();  //!< Creates model matrix from private members
  void initialise_keymap();
  bool post_processing();
  mutable ModelMatrix<2> _model_matrix;
  bool _matrix_is_stored;
  typedef RegisteredParsingObject<PatlakPlot,KineticModel> base_type;
};

END_NAMESPACE_STIR

#endif //__stir_modelling_PatlakPlot_H__
