/*
    Copyright (C) 2020, University College London
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
  \ingroup recon_buildblock
  \brief Declaration of class stir::Hessian_row_sum

  \author Robert Twyman
  \author Kris Thielemans
*/

#ifndef STIR_HESSIAN_ROW_SUM_H
#define STIR_HESSIAN_ROW_SUM_H

#include "stir/info.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.h"

START_NAMESPACE_STIR

class Succeeded;

/*! \ingroup recon_buildblock
  \brief Implementations of the two spatially variant penalty strength (kappa) computation methods propose by Tsai et.al (2020).

  Two methods of computing the kappa images:
    - use an approximation of the Hessian of the log-likelihood
      \code
        kappa = sqrt[ backproj( forwproj(ones) / y) ]
      \endcode
    - use a current image estimate to compute the hessian of the objective function
    \code
      kappa = sqrt[ backproj( (y / (forwproj(lambda) + b)^2 ) * forwproj(ones)) +
                beta * prior.Hessian_times_input(lambda, ones)]
    \endcode
  where y is the measured data, lambda the current image estimate, ones an image of ones.

  For more details, see: Tsai, Y.-J., Schramm, G., Ahn, S., Bousse, A., Arridge, S., Nuyts, J., Hutton, B. F.,
  Stearns, C. W., &  Thielemans, K. (2020). <i>Benefits of Using a Spatially-Variant Penalty Strength With Anatomical
  Priors in PET Reconstruction</i>. IEEE Transactions on Medical Imaging, 39(1), 11–22.
  https://doi.org/10.1109/TMI.2019.2913889
*/
template <typename TargetT>
class Hessian_row_sum:
        public ParsingObject
{
public:
    Hessian_row_sum();

    //! The main function to call to compute spatially variant penalty strength images
    //! Loads the input image from file, sets up, compute Hessian row sum (with input = ones), sqrts the result,
    //! and saves it.
    void process_data();

    //! get and set methods for the objective function
    //@{
    GeneralisedObjectiveFunction<TargetT > const& get_objective_function();
    void set_objective_function_sptr(const shared_ptr<GeneralisedObjectiveFunction<TargetT > > &obj_fun);
    //@}

    //! get and set methods for the input image
    //@{
    shared_ptr<TargetT> get_input_image();
    void set_input_image(shared_ptr <TargetT > const& image);
    //@}

    //! get method for returning the kappa_image
    shared_ptr<TargetT> get_output_target_sptr();

    //! This method resets the output target to uniform 0s
    void reset_output_target_sptr(shared_ptr <TargetT > const& image);

    //! get and set methods for use approximate hessian bool
    //@{
    bool get_use_approximate_hessian();
    void set_use_approximate_hessian(bool use_approximate);
    //@}

    //! get and set methods for the compute with penalty bool
    //@{
    bool get_compute_with_penalty();
    void set_compute_with_penalty(bool with_penalty);
    //@}

    //! Computes the objective function Hessian at the current image estimate. Can use the penalty's Hessian
    //! kappa_image_sptr is the
    void compute_Hessian_row_sum();

    //! Computes the approximate Hessian of the objective function. Cannot use penalty's Hessian
    void compute_approximate_Hessian_row_sum();

protected:

private:
    //! Objective function object
    shared_ptr<GeneralisedObjectiveFunction<TargetT> >  objective_function_sptr;

    //! The filename the for the output spatially variant penalty strength
    std::string output_filename;

    //! Used to load an image as a template or current image estimate to compute hessian at
    std::string input_image_filename;

    //! The input image, can be template or current_image_estimate, dependant on use_approximate_hessian
    shared_ptr<TargetT> input_image;

    //! The variable to which kappa computation methods will populate
    shared_ptr<TargetT> output_target_sptr;

    //! Used to toggle the approximate hessian or hessian at current image estimate should be computed
    //! This toggles the usage of input_image.
    //! If true, input_image is used as template image for the back-projection,
    //! else, input_image is used as the current_image_estimate
    bool use_approximate_hessian;

    //! When computing the hessian of the objective function, include the penalty term or not.
    //! Does not work with use_approximate_hessian as priors do not have an approximate method
    bool compute_with_penalty;

    //! Method to save images
    shared_ptr<OutputFileFormat<TargetT> > output_file_format_sptr;

    //! used to check acceptable parameter ranges, etc...
    bool post_processing();
    void initialise_keymap();
    void set_defaults();
};

END_NAMESPACE_STIR
#endif //STIR_HESSIAN_ROW_SUM_H
