/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class stir::SqrtHessianRowSum

  \author Robert Twyman
  \author Kris Thielemans
*/

#ifndef STIR_SQRTHESSIANROWSUM_H
#define STIR_SQRTHESSIANROWSUM_H

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
  \brief Implementations of two square root Hessian row sum computation methods, as proposed by Tsai et al. (2020).

  Two methods of computing the sqrt of the Hessian row sum images:
    - use an approximation of the Hessian of the log-likelihood from only the measured data
      \code
        sqrt[ backproj( forwproj(ones) / y) ]
      \endcode
    - use a current image estimate to compute the Hessian of the objective function (and optionally Hessian of the prior)
    \code
      sqrt[ backproj( (y / (forwproj(lambda) + b)^2 ) * forwproj(ones)) +
                beta * prior.Hessian_times_input(lambda, ones)]
    \endcode
  where y is the measured (input) data, b is the corrections (additive sinogram), lambda is the current image estimate,
  and ones is an uniform array of ones.

  For more details, see: Tsai, Y.-J., Schramm, G., Ahn, S., Bousse, A., Arridge, S., Nuyts, J., Hutton, B. F.,
  Stearns, C. W., &  Thielemans, K. (2020). <i>Benefits of Using a Spatially-Variant Penalty Strength With Anatomical
  Priors in PET Reconstruction</i>. IEEE Transactions on Medical Imaging, 39(1), 11–22.
  https://doi.org/10.1109/TMI.2019.2913889
*/
template <typename TargetT>
class SqrtHessianRowSum:
        public ParsingObject
{
public:
    //! Default constructor
    /*! calls set_defaults().*/
    SqrtHessianRowSum();
    explicit SqrtHessianRowSum(const std::string&);
    
    //! sets default values
    /*! Sets \c use_approximate_hessian to \c true and \c compute_with_penalty to \c false
    */
    void set_defaults();

    //! The main function to compute and save the sqrt of the Hessian row sum volume
    /*! Different Hessian row sum methods can be used, see compute_Hessian_row_sum() and
          compute_approximate_Hessian_row_sum().*/
    void process_data();

    //! \name get and set methods for the objective function sptr
    //@{
    GeneralisedObjectiveFunction<TargetT > const& get_objective_function_sptr();
    void set_objective_function_sptr(const shared_ptr<GeneralisedObjectiveFunction<TargetT > > &obj_fun);
    //@}

    //! \name get and set methods for the input image
    //@{
    shared_ptr<TargetT> get_input_image_sptr();
    void set_input_image_sptr(shared_ptr <TargetT > const& image);
    //@}

    //! get method for returning the sqrt row sum image
    shared_ptr<TargetT> get_output_target_sptr();

    void set_up();

    //! \name get and set methods for use approximate hessian bool
    //@{
    bool get_use_approximate_hessian() const;
    void set_use_approximate_hessian(bool use_approximate);
    //@}

    //! \name get and set methods for the compute with penalty bool
    //@{
    bool get_compute_with_penalty() const;
    void set_compute_with_penalty(bool with_penalty);
    //@}

    //! Computes the objective function Hessian row sum at the current image estimate.
    //! Can compute the penalty's Hessian if it exists for the selected prior.
    void compute_Hessian_row_sum();

    //! Computes the approximate Hessian of the objective function.
    //! Cannot use penalty's approximate Hessian, see compute_with_penalty
    void compute_approximate_Hessian_row_sum();

protected:

private:
    bool _already_setup = false;

    //! Objective function object
    shared_ptr<GeneralisedObjectiveFunction<TargetT> >  objective_function_sptr;

    //! The filename the for the output sqrt row sum image
    std::string output_filename;

    //! Used to load an image as a template or current image estimate to compute sqrt row sum
    std::string input_image_filename;

    //! The input image, can be template or current_image_estimate, dependant on which sqrt row sum method used
    shared_ptr<TargetT> input_image_sptr;

    //! The output image that the row sum computation methods will populate
    shared_ptr<TargetT> output_target_sptr;

    //! Used to toggle which of the two row sum methods will be utilised.
    //! This toggles the usage of input_image_sptr.
    //! If true, input_image_sptr is only used as a template for the output back-projection,
    //! else, input_image_sptr is used as the current_image_estimate
    bool use_approximate_hessian;

    //! When computing the hessian row sum of the objective function, include the penalty term or not.
    //! Does not work with use_approximate_hessian as priors do not have an approximate method.
    bool compute_with_penalty;

    //! File-format to save images
    shared_ptr<OutputFileFormat<TargetT> > output_file_format_sptr;

    //! used to check acceptable parameter ranges, etc...
    bool post_processing();
    void initialise_keymap();
};

END_NAMESPACE_STIR
#endif //STIR_SQRTHESSIANROWSUM_H
