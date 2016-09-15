#ifndef __stir_scatter_ScatterEstimationByBin_H__
#define __stir_scatter_ScatterEstimationByBin_H__

/*
    Copyright (C) 2016 University College London
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
  \ingroup scatter
  \brief Definition of class stir::ScatterEstimationByBin.
  
  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"
#include "stir/ParsingObject.h"
#include "stir/numerics/BSplines.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/recon_buildblock/Reconstruction.h"

#include "stir/scatter/ScatterSimulation.h"
#include "stir/recon_buildblock/ChainedBinNormalisation.h"

#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/AnalyticReconstruction.h"

#include "stir/PostFiltering.h"

START_NAMESPACE_STIR

//!
//! \bief mask_parameters
//! \details A struct to hold the parameters for
//! masking.
//! \todo Maybe It could be moved it to STIR math.
//!
typedef struct mask_parameters
{
    float max_threshold;
    float add_scalar;
    float min_threshold;
    float times_scalar;
};

/*!
  \ingroup scatter
  \brief Estimate the scatter probability using a model-based approach
*/

class ScatterEstimationByBin : public ParsingObject
{
public:
    //! upsample coarse scatter estimate and fit it to tails of the emission data
    /*! Current procedure:
    1. interpolate segment 0 of \a scatter_proj_data to size of segment 0 of \a emission_proj_data
    2. inverseSSRB to create oblique segments
    3. find scale factors with get_scale_factors_per_sinogram()
    4. apply thresholds
    5. filter scale-factors in axial direction (independently for every segment)
    6. apply scale factors using scale_sinograms()
  */
    static void
    upsample_and_fit_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                      const  ProjData& emission_proj_data,
                                      const ProjData& scatter_proj_data,
                                      const BinNormalisation& scatter_normalisation,
                                      const ProjData& weights_proj_data,
                                      const float min_scale_factor,
                                      const float max_scale_factor,
                                      const unsigned half_filter_width,
                                      BSpline::BSplineType spline_type,
                                      const bool remove_interleaving = true);


    //! Default constructor (calls set_defaults())
    ScatterEstimationByBin();

    virtual Succeeded process_data();

    virtual Succeeded reconstruct_iterative(int,
                                            shared_ptr<DiscretisedDensity<3, float> >&);

    virtual Succeeded reconstruct_analytic();

    // TODO write_log can't be const because parameter_info isn't const
    virtual void
    write_log(const double simulation_time,
              const float total_scatter);

    bool run_debug_mode;

protected:
    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    //!
    //! \brief set_up
    //! \return
    //! \details This function will take care most of the initialisation needed:
    //! <ul>
    //! <li> Procedure:
    //!     <ol>
    //!     <li> Check if debug mode and activate all export flags.
    //!     <li> Load the input_projdata_3d_sptr and perform SSRB
    //!     <li> If recompute_initial_activity_image is set and has a filename
    //!          then load it. Else, create and initialise the activity image to 1s.
    //!     <li> Initialise (partially for the moment) the reconstruction method:
    //!          Parses the input par file.
    //!     <li> Load the attenuation image, and perform basic checks.
    //!     <li>
    //!     </ol>
    //! </ul>
    virtual Succeeded set_up();

    //!
    //! \brief set_up_iterative
    //! \return
    Succeeded set_up_iterative(IterativeReconstruction<DiscretisedDensity<3, float> > * arg);

    //!
    //! \brief set_up_initialise_analytic
    //! \return
    Succeeded set_up_analytic();

    //! Recompute or load the mask image.
    bool recompute_mask_atten_image;
    //! If set the mask projdata will be recomputed
    bool recompute_mask_projdata;
    //! If set to 1 the attenuation coefficients are going to
    //! be recalculated.
    bool recompute_atten_projdata;
    //! If set then the initial activity estimate will be recomputed
    //! and stored if a name is provided.
    bool recompute_initial_activity_image;


    //!
    //! \brief reconstruction_method_sptr
    //! \details The reconsturction which is going to be used for the scatter simulation
    //! and the intial activity image (if recompute set). It can be defined in the same
    //! parameters file as the scatter parameters or to an external via the
    //! reconstruction_template_par_filename
    shared_ptr < Reconstruction < DiscretisedDensity < 3, float > > >
    reconstruction_template_sptr;

    //!
    //! \brief activity_image_sptr
    //! \details Initially with is the reconstructed activity image, but during the scatter
    //! estimation it with actually hold the iterative estimates.
    //! Therefore the nane might change later.
    shared_ptr<DiscretisedDensity < 3, float > > activity_image_sptr;

    shared_ptr<DiscretisedDensity < 3, float > > activity_image_lowres_sptr;

    //!
    //! \brief atten_image_sptr
    //!
    shared_ptr<DiscretisedDensity < 3, float > > atten_image_sptr;

    shared_ptr<DiscretisedDensity <3, float> > atten_image_lowres_sptr;

    shared_ptr<ChainedBinNormalisation>  multiplicative_data_2d_sptr;

    //! \details shared pointer to projection data info for the 2D data - after SSRB
    shared_ptr < ProjDataInfo > proj_data_info_2d_sptr;

    //! The 3D attenuation projdata are used only in
    //! the end of the scatter estimation.
    shared_ptr< ProjData > atten_projdata_sptr;
    //!
    shared_ptr<ProjData> atten_projdata_2d_sptr;
    //! Normalisation proj_data 3D
    shared_ptr<ProjData> norm_projdata_sptr;
    //! Normalisation projdata after SSRB
    shared_ptr<ProjData> norm_projdata_2d_sptr;
    //! Mask proj_data
    shared_ptr<ProjData> mask_projdata_sptr;
    //! Scatter Estimation proj_data
    shared_ptr<ProjData> output_projdata_sptr;
    //! The full 3D projdata are used for the calculation of the 2D
    //! and later for the upsampling back to 3D.
    shared_ptr<ProjData> input_projdata_sptr;
    //! The 2D projdata are used for the scatter estimation.
    shared_ptr<ProjData> input_projdata_2d_sptr;
    //! Original Background projdata
    shared_ptr<ProjData> back_projdata_sptr;
    //! Background projection data after SSRB
    shared_ptr<ProjData> back_projdata_2d_sptr;


    //! Filename of the initial activity image.
    std::string initial_activity_image_filename;
    //! Filename of mask image
    std::string mask_atten_image_filename;
    //! Postfilter parameter file to be used in mask calculation
    std::string mask_postfilter_filename;
    //! Filename of mask's projdata
    std::string mask_projdata_filename;
    //! Filename of background projdata
    std::string back_projdata_filename;
    //! Filename of normalisation factors
    std::string norm_projdata_filename;
    //! Paraameter file for the tail fitting.
    std::string tail_mask_par_filename;
    //! Filename of the measured emission 3D data.
    std::string input_projdata_filename;
    //! This is the image file name with the anatomic information.
    std::string atten_image_filename;
    //! The filename for the parameters file of the reconstruction method.
    std::string recon_template_par_filename;
    //! The file name for the attenuation coefficients.
    //! If they are to be recalculated they will be stored here, if set.
    std::string atten_coeff_filename;

    //!
    shared_ptr < DiscretisedDensity < 3, float >  > mask_atten_image_sptr;

    //! \details the set of parameters to mask the attenuation image
    mask_parameters mask_attenuation_image;

    //    std::string output_projdata_filename;
private:

    //! \details A helper function to reduce the size of set_up().ß
    Succeeded ffw_project_mask_image();
    //! \details A helper function to reduce the size of set_up().
    void apply_mask_in_place(shared_ptr<DiscretisedDensity<3, float> >&,
                             const mask_parameters&);

    //! \details Average the two first activity images 0 and 1.
    bool do_average_at_2;
    //! \details Used for convinience. It is initialised on post_processing.
    bool iterative_method;
    //!
    bool remove_interleaving;
    //! Save all scatter simulated sinograms
    bool export_scatter_estimates_of_each_iteration;
    //! Export SSRB sinograms
    bool export_2d_projdata;

    //! Parameter file for scatter simulation
    //! \warning Values in this file could be overridden.
    std::string scatter_sim_par_filename;

    //! \details Class which will implement the scatter simulation.
    shared_ptr < ScatterSimulation > scatter_simulation_sptr;

    shared_ptr<ProjData> multimulti2d;

    shared_ptr<PostFiltering <DiscretisedDensity<3,float> > > filter_sptr;

    //! \details The number of iterations the scatter estimation will perform.
    //! Default = 5.
    int num_scatter_iterations;

    //! Default value = 100
    float max_scale_value;
    //! Default value = 0.4
    float min_scale_value;
    //!
    float half_filter_width;
    //! Ouput file name prefix
    std::string o_scatter_estimate_prefix;
};

END_NAMESPACE_STIR
#endif
