#ifndef __stir_scatter_ScatterEstimation_H__
#define __stir_scatter_ScatterEstimation_H__

/*
    Copyright (C) 2018 - 2019 University of Hull
    Copyright (C) 2016,2020 University College London
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
  \brief Definition of class stir::ScatterEstimation.
  
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

#include "stir/stir_math.h"
#include "stir/FilePath.h"

START_NAMESPACE_STIR

template <class TargetT> class PostFiltering;

//! A struct to hold the parameters for image masking.
struct MaskingParameters
{
  float min_threshold;
  //! filter parameter file to be used in mask calculation
  std::string filter_filename;
  //! filter to apply before thresholding
  shared_ptr<PostFiltering <DiscretisedDensity<3,float> > > filter_sptr;
};

/*!
  \ingroup scatter
  \brief Estimate the scatter probability using a model-based approach
  \author Nikos Efthimiou
  \author Kris Thielemans

  This is an implementation of Watson's SSS iterative algorithm to estimate scatter.
  Simulation is normally at low resolution, ideally reconstruction as well (to save time).
  The code also does tail-fitting.

  A particular feature is that the first iterations happen in 2D (after SSRB of all the data).
  Currently only the last simulation is in 3D. This could be changed later.

  \todo The code should throw an error if 2D input data are loaded.
  It should just deactivate the final upsampling.
  \todo Currently FBP reconstruction is not working and just throws an error.
  \todo This code needs far more documentation.
*/

class ScatterEstimation: public ParsingObject
{
public:
    //! upsample coarse scatter estimate and fit it to tails of the emission data
    /*! Current procedure:
    1. interpolate segment 0 of \a scatter_proj_data to size of segment 0 of \a emission_proj_data
    2. inverseSSRB to create oblique segments
    3. undo normalisation (as measured data is not normalised)
    4. find scale factors with get_scale_factors_per_sinogram()
    5. apply thresholds
    6. filter scale-factors in axial direction (independently for every segment)
    7. apply scale factors using scale_sinograms()
  */
    static void
    upsample_and_fit_scatter_estimate(ProjData& scaled_scatter_proj_data,
                                      const  ProjData& emission_proj_data,
                                      const ProjData& scatter_proj_data,
                                      BinNormalisation& scatter_normalisation,
                                      const ProjData& weights_proj_data,
                                      const float min_scale_factor,
                                      const float max_scale_factor,
                                      const unsigned half_filter_width,
                                      BSpline::BSplineType spline_type,
                                      const bool remove_interleaving = true);


    //! Default constructor (calls set_defaults())
    ScatterEstimation();
    //! Overloaded constructor with parameter file and initialisation
    explicit ScatterEstimation(const std::string& parameter_filename);

    //! Full process_data which performs set_up() before beginning
    virtual Succeeded process_data();

    //! Get current scatter estimate
    shared_ptr<ProjData> get_output() const;

    //!
    //! \brief set_up
    //! \return
    //! \details This function will take care most of the initialisation needed:
    //! <ul>
    //! <li> Procedure:
    //!     <ol>
    //!     <li> Check if debug mode and activate all export flags.
    //!     <li> Load the input_projdata_3d_sptr and perform SSRB
    //!     <li> Initialise (partially for the moment) the reconstruction method:
    //!          Parses the input par file.
    //!     <li> Load Normalisation data and perform SSRB
    //!     <li> Load the background data (randoms) and do normalisation (to get the additive data)
    //!     </ol>
    //! </ul>
    virtual Succeeded set_up();

    // Set functions
    //! Set the input projdata.
    inline void set_input_proj_data_sptr(const shared_ptr<ProjData>);
    //! Set the reconstruction method for the scatter estimation
    inline void set_reconstruction_method_sptr(const shared_ptr<Reconstruction < DiscretisedDensity < 3, float > > >);
    //! Set the full resolution attenuation image.
    inline void set_attenuation_image_sptr(const shared_ptr<const DiscretisedDensity<3, float > > );
    //!
    inline void set_attenuation_correction_proj_data_sptr(const shared_ptr<ProjData>);
    //!
    inline void set_normalisation_proj_data_sptr(const shared_ptr<ProjData>);
    //!
    inline void set_background_proj_data_sptr(const shared_ptr<ProjData>);
    //!
    inline void set_initial_activity_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> >);

    inline void set_mask_image_sptr(const shared_ptr<const DiscretisedDensity<3, float> >);
    //! set mask for tail-fitting
    /*! \c arg will not be modified */
    inline void set_mask_proj_data_sptr(const shared_ptr<ProjData> arg);

    inline void set_scatter_simulation_method_sptr(const shared_ptr<ScatterSimulation>);
    //! Set the zoom factor in the XY plane for the downsampling of the activity and attenuation image.
    inline void set_zoom_xy(float);
    //! Set the zoom factor in the Z axis for the downsampling of the activity and attenuation image.
    inline void set_zoom_z(float);


    // Get functions
    //! Get the number of iteration for the scatter estimation
    int get_iterations_num() const;

protected:
    //! All recomputes_** will default true
    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    //! Recompute or load the mask image.
    bool recompute_mask_image;
    //! If set the mask projdata will be recomputed
    bool recompute_mask_projdata;
    //! If set to 1 the attenuation coefficients are going to
    //! be recalculated.
    bool recompute_atten_projdata;

    //! This is the reconstruction object which is going to be used for the scatter estimation
    //! and the calculation of the initial activity image (if recompute set). It can be defined in the same
    //! parameters file as the scatter parameters or in an external defined in the
    //! reconstruction_template_par_filename
    shared_ptr < Reconstruction < DiscretisedDensity < 3, float > > >
      reconstruction_template_sptr;
    //! The current activity estimate.
    shared_ptr<DiscretisedDensity < 3, float > > current_activity_image_sptr;
    //! Image with attenuation values.
    shared_ptr<const DiscretisedDensity < 3, float > > atten_image_sptr;
    //! ((1/SSRB(1/norm3D)) * SSRB(atten)). Through out the code we set as first the norm
    //! and second the atten.
    shared_ptr<BinNormalisation>  multiplicative_binnorm_2d_sptr;
    //! (norm * atten) in 3D. Through out the code we set as first the norm
    //! and second the atten.
    shared_ptr<BinNormalisation>  multiplicative_binnorm_sptr;
    //! Mask proj_data
    shared_ptr<ProjData> mask_projdata_sptr;
    //! The full 3D projdata are used for the calculation of the 2D
    //! and later for the upsampling back to 3D.
    shared_ptr<ProjData> input_projdata_sptr;
    //! The 2D projdata are used for the scatter estimation.
    shared_ptr<ProjData> input_projdata_2d_sptr;
    //! Additive projection data after SSRB -- Randoms
    shared_ptr<ProjData> add_projdata_2d_sptr;
    //! Prompts - randoms
    shared_ptr<ProjData> data_to_fit_projdata_sptr;

    shared_ptr<ProjData> add_projdata_sptr;
    //! (Additive + Scatter Estimate) * Mult in 2D
    shared_ptr<ProjData> back_projdata_2d_sptr;
    //! Initially this points to the un-normalised randoms.
    shared_ptr<ProjData> back_projdata_sptr;
    //! Filename of mask image
    std::string mask_image_filename;
    //! Filename of mask's projdata
    std::string mask_projdata_filename;
    //! Filename of background projdata
    std::string back_projdata_filename;
    //! Filename of normalisation factors
    std::string norm_coeff_filename;
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

    //! \details the set of parameters to mask the attenuation image
    MaskingParameters masking_parameters;
    //! \details The number of iterations the scatter estimation will perform.
    //! Default = 5.
    int num_scatter_iterations;
    //! Output file name prefix
    std::string output_scatter_estimate_prefix;

    std::string output_additive_estimate_prefix;
private:
    //! variable for storing current scatter estimate
    shared_ptr<ProjData> scatter_estimate_sptr;
    
    //! variable storing the mask image
    shared_ptr < const DiscretisedDensity < 3, float >  > mask_image_sptr;

    //! \brief set_up iterative reconstruction
    Succeeded set_up_iterative(shared_ptr<IterativeReconstruction<DiscretisedDensity<3, float> > > arg);

    //! \brief set_up analytic reconstruction
    Succeeded set_up_analytic();

    //! \details A helper function to reduce the size of set_up().
    Succeeded project_mask_image();

    //! reconstruct image with current scatter estimate (iteratively)
    /*! \a scat_iter is used for determining the filename for saving */
    void reconstruct_iterative(int scat_iter, shared_ptr<DiscretisedDensity<3, float> >& output_sptr);

    //! reconstruct image with current scatter estimate (analytic reconstruction)
    /*! \a scat_iter is used for determining the filename for saving */
    void reconstruct_analytic(int scat_iter, shared_ptr<DiscretisedDensity<3, float> > & output_sptr);

    //! \details Find a mask by thresholding etc
    static void apply_mask_in_place(DiscretisedDensity<3, float> &,
				    const MaskingParameters&);

    void add_proj_data(ProjData&, const ProjData&);

    void subtract_proj_data(ProjData&, const ProjData&);

    void apply_to_proj_data(ProjData& , const pow_times_add&);

    //! extract the normalisation component of a combined norm
    shared_ptr<BinNormalisation>
      get_normalisation_object_sptr(const shared_ptr<BinNormalisation>& combined_norm_sptr);

    //! Returns a shared pointer to a new ProjData. If we run in run_debug_mode and
    //! the extras_path has been set, then it will be a ProjDataInterfile, otherwise it will be a ProjDataInMemory.
    shared_ptr<ProjData> create_new_proj_data(const std::string& filename,
					      const shared_ptr<const ExamInfo> exam_info_sptr,
					      const shared_ptr<const ProjDataInfo> proj_data_info_sptr) const;

    //! \details Average the two first activity images 0 and 1 (defaults to \c true)
    bool do_average_at_2;
    //! for upsampling (defaults to \c true)
    bool remove_interleaving;
    //! Save all scatter simulated sinograms
    bool export_scatter_estimates_of_each_iteration;
    //! Run the process in 2D by SSRB the 3D sinograms
    bool run_in_2d_projdata;
    //! This bool will allow the ScatterEstimation to override the value of
    //! the density image set in ScatterSimulation par file (defaults to \c true)
    bool override_density_image;
    //! This will over-ride the scanner template in scatter sinogram simulation (defaults to \c true)
    bool override_scanner_template;
    //! In debug mode a lot of extra files are going to be saved in the disk.
    bool run_debug_mode;
    //! Parameter file for scatter simulation
    //! \warning Values in this file could be overridden.
    std::string scatter_sim_par_filename;
    //! \details Class which will implement the scatter simulation.
    shared_ptr < ScatterSimulation > scatter_simulation_sptr;
    //! This path is used in the debug mode to store all the intermediate files, as they are many.
    FilePath extras_path;

    //! Default value = 100
    float max_scale_value;
    //! Default value = 0.4
    float min_scale_value;

    bool downsample_scanner_bool;
    //!
    unsigned int half_filter_width;

    //! \details internal variable set to \c true when using iterative reconstruction
    bool iterative_method;
};

END_NAMESPACE_STIR
#include "stir/scatter/ScatterEstimation.inl"
#endif
