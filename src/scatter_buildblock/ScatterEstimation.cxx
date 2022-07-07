/*
  Copyright (C) 2018,2019,2020 University College London
  Copyright (C) 2018-2019, University of Hull
  Copyright (C) 2022 National Physical Laboratory
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup scatter
  \brief Implementation of most functions in stir::ScatterEstimation

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Daniel Deidda
  \author Markus Jehl
*/
#include "stir/scatter/ScatterEstimation.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/recon_buildblock/ChainedBinNormalisation.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/SSRB.h"
#include "stir/DataProcessor.h"
#include "stir/PostFiltering.h"
#include "stir/scatter/CreateTailMaskFromACFs.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/zoom.h"
#include "stir/ZoomOptions.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/ArrayFunction.h"
#include "stir/NumericInfo.h"
#include "stir/SegmentByView.h"
#include "stir/VoxelsOnCartesianGrid.h"

// The calculation of the attenuation coefficients
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"

START_NAMESPACE_STIR

void
ScatterEstimation::
set_defaults()
{
    this->_already_setup = false;
    this->scatter_simulation_sptr.reset(new SingleScatterSimulation);
    this->recompute_atten_projdata = true;
    this->recompute_mask_image = true;
    {
      // image masking
      this->masking_parameters.min_threshold = .003F;
      shared_ptr<SeparableGaussianImageFilter<float> > filter_sptr(new SeparableGaussianImageFilter<float>);
      filter_sptr->set_fwhms(make_coordinate(15.F, 20.F, 20.F));
      this->masking_parameters.filter_sptr.reset(new PostFiltering<DiscretisedDensity < 3, float > >);
      this->masking_parameters.filter_sptr->set_filter_sptr(filter_sptr);
    }
    this->recompute_mask_projdata = true;
    this->run_in_2d_projdata = true;
    this->do_average_at_2 = true;
    this->export_scatter_estimates_of_each_iteration = false;
    this->run_debug_mode = false;
    this->override_scanner_template = true;
    this->override_density_image = true;
    this->downsample_scanner_bool = true;
    this->remove_interleaving = true;
    this->atten_image_filename = "";
    this->atten_coeff_filename = "";
    this->norm_3d_sptr.reset();
    this->multiplicative_binnorm_sptr.reset();
    this->output_scatter_estimate_prefix = "";
    this->output_additive_estimate_prefix = "";
    this->num_scatter_iterations = 5;
    this->min_scale_value = 0.4f;
    this->max_scale_value = 100.f;
    this->half_filter_width = 3;
}

void
ScatterEstimation::
initialise_keymap()
{
    this->parser.add_start_key("Scatter Estimation Parameters");
    this->parser.add_stop_key("end Scatter Estimation Parameters");

    this->parser.add_key("run in debug mode",
                         &this->run_debug_mode);
    this->parser.add_key("input file",
                         &this->input_projdata_filename);
    this->parser.add_key("attenuation image filename",
                         &this->atten_image_filename);

    // MASK parameters
    this->parser.add_key("recompute mask image",
                         &this->recompute_mask_image);
    this->parser.add_key("mask image filename",
                         &this->mask_image_filename);
    this->parser.add_key("mask attenuation image filter filename",
                         &this->masking_parameters.filter_filename);
    this->parser.add_key("mask attenuation image min threshold",
                         &this->masking_parameters.min_threshold);
    this->parser.add_key("recompute mask projdata",
                         &this->recompute_mask_projdata);
    this->parser.add_key("mask projdata filename",
                         &this->mask_projdata_filename);
    this->parser.add_key("tail fitting parameter filename",
                         &this->tail_mask_par_filename);
    // END MASK
    this->parser.add_key("background projdata filename",
                         &this->back_projdata_filename);
    this->parser.add_parsing_key("Normalisation type",
                         &this->norm_3d_sptr);
    this->parser.add_key("attenuation correction factors filename",
                         &this->atten_coeff_filename);
    this->parser.add_parsing_key("Bin Normalisation type",
                         &this->multiplicative_binnorm_sptr);

    // RECONSTRUCTION RELATED
    this->parser.add_key("reconstruction parameter filename",
                         &this->recon_template_par_filename);
    this->parser.add_parsing_key("reconstruction type",
                                 &this->reconstruction_template_sptr);
    // END RECONSTRUCTION RELATED

    this->parser.add_key("number of scatter iterations",
                         &this->num_scatter_iterations);
    //Scatter simulation
    this->parser.add_parsing_key("Scatter Simulation type",
                                 &this->scatter_simulation_sptr);
    this->parser.add_key("scatter simulation parameter filename",
                         &this->scatter_sim_par_filename);
    this->parser.add_key("use scanner downsampling in scatter simulation",
                         &this->downsample_scanner_bool);

    this->parser.add_key("override attenuation image",
                         &this->override_density_image);
    this->parser.add_key("override scanner template",
                         &this->override_scanner_template);

    // END Scatter simulation

    this->parser.add_key("export scatter estimates of each iteration",
                         &this->export_scatter_estimates_of_each_iteration);
    this->parser.add_key("output scatter estimate name prefix",
                         &this->output_scatter_estimate_prefix);
    this->parser.add_key("output additive estimate name prefix",
                         &this->output_additive_estimate_prefix);
    this->parser.add_key("do average at 2",
                         &this->do_average_at_2);
    this->parser.add_key("maximum scatter scaling factor",
                         &this->max_scale_value);
    this->parser.add_key("minimum scatter scaling factor",
                         &this->min_scale_value);
    this->parser.add_key("upsampling half filter width",
                         &this->half_filter_width);
    this->parser.add_key("remove interleaving before upsampling",
                         &this->remove_interleaving);
    this->parser.add_key("run in 2d projdata",
                         &this->run_in_2d_projdata);
}

ScatterEstimation::
ScatterEstimation()
{
    this->set_defaults();
}

ScatterEstimation::
ScatterEstimation(const std::string& parameter_filename)
{
  this->set_defaults();
  if (!this->parse(parameter_filename.c_str()))
    {
      error("ScatterEstimation: Error parsing input file %s. Aborting.", parameter_filename.c_str());
    }
}


shared_ptr<ProjData> 
ScatterEstimation::
make_2D_projdata_sptr(const shared_ptr<ProjData> in_3d_sptr)
{
    shared_ptr<ProjData> out_2d_sptr;
    if (in_3d_sptr->get_proj_data_info_sptr()->get_scanner_sptr()->get_scanner_geometry()=="Cylindrical")
    {
        shared_ptr<ProjDataInfo> out_info_2d_sptr(SSRB(*in_3d_sptr->get_proj_data_info_sptr(),in_3d_sptr->get_num_segments(), 1, false));
        out_2d_sptr.reset(new ProjDataInMemory(in_3d_sptr->get_exam_info_sptr(),
                                               out_info_2d_sptr));
        
        SSRB(*out_2d_sptr,
             *in_3d_sptr,false);
    }
    else
    { 
        shared_ptr<ProjDataInfo> out_info_2d_sptr(in_3d_sptr->get_proj_data_info_sptr()->create_shared_clone());
        out_info_2d_sptr->reduce_segment_range(0,0);
        out_2d_sptr.reset(new ProjDataInMemory(in_3d_sptr->get_exam_info_sptr(),
                                                                out_info_2d_sptr));
        
        SegmentBySinogram<float> segment=in_3d_sptr->get_segment_by_sinogram(0);
        out_2d_sptr->set_segment(segment);
//        std::cout<<" value "<<out_2d_sptr->get_sinogram(8,0)[0][0]<<std::endl;
    }
    return out_2d_sptr;
}

shared_ptr<ProjData> 
ScatterEstimation::
make_2D_projdata_sptr(const shared_ptr<ProjData> in_3d_sptr, string template_filename)
{
    shared_ptr<ProjData> out_2d_sptr;
    if (in_3d_sptr->get_proj_data_info_sptr()->get_scanner_sptr()->get_scanner_geometry()=="Cylindrical")
    {
        shared_ptr<ProjDataInfo> out_info_2d_sptr(SSRB(*in_3d_sptr->get_proj_data_info_sptr(),in_3d_sptr->get_num_segments(), 1, false));
        out_2d_sptr = create_new_proj_data(template_filename, 
                                           this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                           this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
        
        SSRB(*out_2d_sptr,
             *in_3d_sptr,false);
    }
    else
    { 
        shared_ptr<ProjDataInfo> out_info_2d_sptr(in_3d_sptr->get_proj_data_info_sptr()->create_shared_clone());
        out_info_2d_sptr->reduce_segment_range(0,0);
        out_2d_sptr.reset(new ProjDataInMemory(in_3d_sptr->get_exam_info_sptr(),
                                                                out_info_2d_sptr));
        
        SegmentBySinogram<float> segment=in_3d_sptr->get_segment_by_sinogram(0);
        out_2d_sptr->set_segment(segment);
//        std::cout<<" value "<<out_2d_sptr->get_sinogram(8,0)[0][0]<<std::endl;
    }
    return out_2d_sptr;
}

bool
ScatterEstimation::
post_processing()
{
    if (!this->input_projdata_filename.empty())
      {
        info("ScatterEstimation: Loading input projdata...", 3);
        this->input_projdata_sptr =
          ProjData::read_from_file(this->input_projdata_filename);
      }
    // If the reconstruction_template_sptr is null then, we need to parse it from another
    // file. I prefer this implementation since makes smaller modular files.
    if (!this->recon_template_par_filename.empty())
    {
        KeyParser local_parser;
        local_parser.add_start_key("Reconstruction Parameters");
        local_parser.add_stop_key("End Reconstruction Parameters");
        local_parser.add_parsing_key("reconstruction type", &this->reconstruction_template_sptr);
        if (!local_parser.parse(this->recon_template_par_filename.c_str()))
        {
            warning(boost::format("ScatterEstimation: Error parsing reconstruction parameters file %1%. Aborting.")
                    %this->recon_template_par_filename);
            return true;
        }
    }

    if (!this->atten_image_filename.empty())
      {
        info("ScatterEstimation: Loading attenuation image...", 3);
        this->atten_image_sptr =
          read_from_file<DiscretisedDensity<3,float> >(this->atten_image_filename);
      }
    if (!this->atten_coeff_filename.empty())
      {
        info("ScatterEstimation: Loading attenuation coefficients projdata...", 3);
        shared_ptr<ProjData> atten_coef_sptr =
          ProjData::read_from_file(this->atten_coeff_filename);
        this->set_attenuation_correction_proj_data_sptr(atten_coef_sptr);
      }
    if(!is_null_ptr(multiplicative_binnorm_sptr))
      {
        warning("ScatterEstimation: looks like you set a combined norm via the 'bin normalisation type' keyword\n"
                "This is deprecated and will be removed in a future version (5.0?).\n"
                "Use 'normalisation type' (for the norm factors) and 'attenuation correction factors filename' instead.");
      }

    if (!this->back_projdata_filename.empty())
      {
        info("ScatterEstimation: Loading background projdata...", 3);
        this->back_projdata_sptr =
          ProjData::read_from_file(this->back_projdata_filename);
      }

    //    if(!this->recompute_initial_activity_image ) // This image can be used as a template
    //    {
    //        info("ScatterEstimation: Loading initial activity image ...");
    //        if(this->initial_activity_image_filename.size() > 0 )
    //            this->current_activity_image_lowres_sptr =
    //                read_from_file<DiscretisedDensity<3,float> >(this->initial_activity_image_filename);
    //        else
    //        {
    //            warning("ScatterEstimation: Recompute initial activity image was set to false and"
    //                    "no filename was set. Aborting.");
    //            return true;
    //        }
    //    }

    if(!this->masking_parameters.filter_filename.empty())
    {
        this->masking_parameters.filter_sptr.reset(new PostFiltering <DiscretisedDensity<3,float> >);

        if(!masking_parameters.filter_sptr->parse(this->masking_parameters.filter_filename.c_str()))
        {
            warning(boost::format("ScatterEstimation: Error parsing post filter parameters file %1%. Aborting.")
                    %this->masking_parameters.filter_filename);
            return true;
        }
    }

    if (!this->scatter_sim_par_filename.empty())
      {
        info ("ScatterEstimation: Initialising Scatter Simulation ...", 3);
        // Parse locally
        {
          KeyParser local_parser;
          local_parser.add_start_key("Scatter Simulation Parameters");
          local_parser.add_stop_key("End Scatter Simulation Parameters");
          local_parser.add_parsing_key("Scatter Simulation type", &this->scatter_simulation_sptr);
          if (!local_parser.parse(this->scatter_sim_par_filename.c_str()))
            error("ScatterEstimation: Error parsing scatter simulation parameters.");
        }
      }

    // There is no output in this case
    if (this->output_scatter_estimate_prefix.empty() && this->output_additive_estimate_prefix.empty())
      {
        // This is ok when running from Python or so, but not when running from the command line.
        // As we don't know, we just write a warning
        warning("ScatterEstimation: no filename prefix set for either the scatter estimate or the additive.\n"
                "This is probably not what you want.");
      }

    if(!this->recompute_mask_projdata)
      {
        if (!this->mask_projdata_filename.empty())
          this->mask_projdata_sptr =
            ProjData::read_from_file(this->mask_projdata_filename);
      }
    else
      {
        if (!this->recompute_mask_image && !this->mask_image_filename.empty())
          this->mask_image_sptr =
            read_from_file<DiscretisedDensity<3, float> >(this->mask_image_filename);
      }

    return false;
}

shared_ptr<ProjData> 
ScatterEstimation::get_output() const
{
    return scatter_estimate_sptr;
}

#if STIR_VERSION < 050000
void ScatterEstimation::set_input_data(const shared_ptr<ProjData>& data)
{
  this->set_input_proj_data_sptr(data);
}
#else
void ScatterEstimation::set_input_data(const shared_ptr<ExamData>& data)
{
  // C++-11
  auto sptr = std::dynamic_pointer_cast<ProjData>(data);
  if (!sptr)
    error("ScatterEstimation can only accept ProjData at the moment");

  this->set_input_proj_data_sptr(sptr);
}
#endif

shared_ptr<const ProjData> ScatterEstimation::get_input_data() const
{
  return this->input_projdata_sptr;
}

shared_ptr<const DiscretisedDensity<3,float> >
ScatterEstimation::get_estimated_activity_image_sptr() const
{
  return this->current_activity_image_sptr;
}

void ScatterEstimation::set_output_scatter_estimate_prefix(const std::string& arg)
{
  this->output_scatter_estimate_prefix = arg;
}

void ScatterEstimation::set_export_scatter_estimates_of_each_iteration(bool arg)
{
  this->export_scatter_estimates_of_each_iteration = arg;
}

void ScatterEstimation::set_max_scale_value(float value)
{ this->max_scale_value = value; }

void ScatterEstimation::set_min_scale_value(float value)
{ this->min_scale_value = value; }

void ScatterEstimation::set_mask_projdata_filename(std::string name)
{ this->mask_projdata_filename = name; }

void ScatterEstimation::set_mask_image_filename(std::string name)
{ this->mask_image_filename = name; }

void ScatterEstimation::set_output_additive_estimate_prefix(std::string name)    
{ this->output_additive_estimate_prefix = name; }

void ScatterEstimation::set_run_debug_mode(bool debug)
{ this->run_debug_mode = debug; }


void
ScatterEstimation::
set_attenuation_correction_proj_data_sptr(const shared_ptr<ProjData> arg)
{
  this->_already_setup = false;
  this->atten_norm_3d_sptr.reset(new BinNormalisationFromProjData(arg));
  this->multiplicative_binnorm_sptr.reset();
}

void
ScatterEstimation::
set_normalisation_sptr(const shared_ptr<BinNormalisation> arg)
{
  this->_already_setup = false;
  this->norm_3d_sptr = arg;
  this->multiplicative_binnorm_sptr.reset();
}

bool ScatterEstimation::already_setup() const
{
  return this->_already_setup;
}

Succeeded
ScatterEstimation::
set_up()
{
    if (this->run_debug_mode)
    {
        info("ScatterEstimation: Debugging mode is activated.");
        this->export_scatter_estimates_of_each_iteration = true;

        // Create extras folder in this location
        FilePath current_full_path(FilePath::get_current_working_directory());
        extras_path = current_full_path.append("extras");
    }

    if (is_null_ptr(this->atten_image_sptr))
      error("ScatterEstimation: No attenuation image has been set. Aborting.");

    if (is_null_ptr(this->input_projdata_sptr))
      error("ScatterEstimation: No input proj_data have been set. Aborting.");

    if (is_null_ptr(this->scatter_simulation_sptr))
      error("ScatterEstimation: Please define a scatter simulation method. Aborting.");

    if (!run_in_2d_projdata)
      error("ScatterEstimation: Currently, only running the estimation in 2D is supported.");

    if(!this->recompute_mask_projdata)
      {
        if (is_null_ptr(this->mask_projdata_sptr))
          error("ScatterEstimation: Please set mask proj_data (or enable computing it)");
      }
    else if (!this->recompute_mask_image)
      {
        if (is_null_ptr(this->mask_image_sptr))
          error("ScatterEstimation: Please set a mask image (or enable computing it)");
      }

   if (this->_already_setup)
      return Succeeded::yes;

   info("Scatter Estimation Parameters (objects that are not set by parsing will not be listed correctly)\n" + this->parameter_info() + "\n\n", 1);

    this->create_multiplicative_binnorm_sptr();
    this->multiplicative_binnorm_sptr->set_up(this->input_projdata_sptr->get_exam_info_sptr(), this->input_projdata_sptr->get_proj_data_info_sptr());

#if 1
    // Calculate the SSRB
    if (input_projdata_sptr->get_num_segments() > 1)
    {
        info("ScatterEstimation: Running SSRB on input data...");
        this->input_projdata_2d_sptr = make_2D_projdata_sptr(this->input_projdata_sptr);
    }
    else
    {
        input_projdata_2d_sptr = input_projdata_sptr;
    }

#else
    {
        std::string tmp_input2D = "./extras/nema_proj_f1g1d0b0.hs_2d.hs";
        this->input_projdata_2d_sptr =
                ProjData::read_from_file(tmp_input2D);
    }
#endif
    info("ScatterEstimation: Setting up reconstruction method ...");

    if(is_null_ptr(this->reconstruction_template_sptr))
    {
	warning("ScatterEstimation: Reconstruction method has not been initialised. Aborting.");
	return Succeeded::no;
    }

    // We have to check which reconstruction method we are going to use ...
    shared_ptr<AnalyticReconstruction> tmp_analytic =
            dynamic_pointer_cast<AnalyticReconstruction >(this->reconstruction_template_sptr);
    shared_ptr<IterativeReconstruction<DiscretisedDensity<3, float> > > tmp_iterative =
            dynamic_pointer_cast<IterativeReconstruction<DiscretisedDensity<3, float> > >(reconstruction_template_sptr);

    if (!is_null_ptr(tmp_analytic))
    {
        if(set_up_analytic() == Succeeded::no)
        {
            warning("ScatterEstimation: set_up_analytic reconstruction failed. Aborting.");
            return Succeeded::no;
        }

        this->iterative_method = false;
    }
    else if (!is_null_ptr(tmp_iterative))
    {
        if(set_up_iterative(tmp_iterative) == Succeeded::no)
        {
            warning("ScatterEstimation: set_up_iterative reconstruction failed. Aborting.");
            return Succeeded::no;
        }

        this->iterative_method = true;
    }
    else
    {
        warning("ScatterEstimation: Failure to detect a method of reconstruction. Aborting.");
        return Succeeded::no;
    }

    if(iterative_method)
        this->current_activity_image_sptr.reset(tmp_iterative->get_initial_data_ptr());

    this->current_activity_image_sptr->fill(1.0);

    //
    // ScatterSimulation
    //

    info("ScatterEstimation: Setting up Scatter Simulation method ...");
    // The images are passed to the simulation.
    // and it will override anything that the ScatterSimulation.par file has done.
    if(this->override_density_image)
    {
        info("ScatterEstimation: Over-riding attenuation image! (The file and settings set in the simulation par file are discarded)");
        this->scatter_simulation_sptr->set_density_image_sptr(this->atten_image_sptr);
    }

    //    if(this->override_initial_activity_image)
    //    {
    //        info("ScatterEstimation: Over-riding activity image! (The file and settings set in the simulation par file are discarded)");
    //        this->scatter_simulation_sptr->set_activity_image_sptr(this->current_activity_image_sptr);
    //    }


    if(this->override_scanner_template)
    {
        info("ScatterEstimation: Over-riding the scanner template! (The file and settings set in the simulation par file are discarded)");
        if (run_in_2d_projdata)
        {
            this->scatter_simulation_sptr->set_template_proj_data_info(*this->input_projdata_2d_sptr->get_proj_data_info_sptr());
            this->scatter_simulation_sptr->set_exam_info(this->input_projdata_2d_sptr->get_exam_info());
        }
        else
        {
            this->scatter_simulation_sptr->set_template_proj_data_info(*this->input_projdata_sptr->get_proj_data_info_sptr());
            this->scatter_simulation_sptr->set_exam_info(this->input_projdata_sptr->get_exam_info());
        }

    }

    if (this->downsample_scanner_bool)
        this->scatter_simulation_sptr->downsample_scanner();

    // Check if Load a mask proj_data

    if(is_null_ptr(this->mask_projdata_sptr) || this->recompute_mask_projdata)
    {
        if(is_null_ptr(this->mask_image_sptr) || this->recompute_mask_image)
        {
            // Applying mask
            // 1. Clone from the original image.
            // 2. Apply to the new clone.
            auto mask_image_ptr(this->atten_image_sptr->clone());
            this->apply_mask_in_place(*mask_image_ptr, this->masking_parameters);
            this->mask_image_sptr.reset(mask_image_ptr);
            if (this->mask_image_filename.size() > 0 )
                OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                        write_to_file(this->mask_image_filename, *this->mask_image_sptr);
        }

        if(project_mask_image() == Succeeded::no)
        {
            warning("ScatterEstimation: Unsuccessful to fwd project the mask image. Aborting.");
            return Succeeded::no;
        }
    }

    this->_already_setup = true;
    info("ScatterEstimation: >>>>Set up finished successfully!!<<<<");
    return Succeeded::yes;
}

Succeeded
ScatterEstimation::
set_up_iterative(shared_ptr<IterativeReconstruction<DiscretisedDensity<3, float> > > iterative_object)
{
    info("ScatterEstimation: Setting up iterative reconstruction ...");

    if(run_in_2d_projdata)
    {
        iterative_object->set_input_data(this->input_projdata_2d_sptr);
    }
    else
        iterative_object->set_input_data(this->input_projdata_sptr);



    //
    // Multiplicative projdata
    //
    shared_ptr<ProjData> tmp_atten_projdata_sptr =
      this->get_attenuation_correction_factors_sptr(this->multiplicative_binnorm_sptr);
    shared_ptr<ProjData> atten_projdata_2d_sptr;
    
    info("ScatterEstimation: 3.Calculating the attenuation projection data...");

    if( tmp_atten_projdata_sptr->get_num_segments() > 1)
    {
        info("ScatterEstimation: Running SSRB on attenuation correction coefficients ...");

        std::string out_filename = "tmp_atten_sino_2d.hs";
        atten_projdata_2d_sptr=make_2D_projdata_sptr(tmp_atten_projdata_sptr, out_filename);
        
    }
    else
    {
        // TODO: this needs more work. -- Setting directly 2D proj_data is buggy right now.
        atten_projdata_2d_sptr = tmp_atten_projdata_sptr;
    }

    info("ScatterEstimation: 4.Calculating the normalisation data...");
    {
      if (run_in_2d_projdata)
        {
          shared_ptr<BinNormalisation> norm3d_sptr =
            this->get_normalisation_object_sptr(this->multiplicative_binnorm_sptr);
          shared_ptr<BinNormalisation> norm_coeff_2d_sptr;

          if ( input_projdata_sptr->get_num_segments() > 1)
            {
              // Some BinNormalisation classes don't know about SSRB.
              // we need to get norm2d=1/SSRB(1/norm3d))

              info("ScatterEstimation: Constructing 2D normalisation coefficients ...");

              std::string out_filename = "tmp_inverted_normdata.hs";
              shared_ptr<ProjData> inv_projdata_3d_sptr = create_new_proj_data(out_filename,
                                                                               this->input_projdata_sptr->get_exam_info_sptr(),
                                                                               this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());
              inv_projdata_3d_sptr->fill(1.f);

              out_filename = "tmp_normdata_2d.hs";
              shared_ptr<ProjData> norm_projdata_2d_sptr = create_new_proj_data(out_filename,
                                                                                this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                                this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
              norm_projdata_2d_sptr->fill(0.f);

              // Essentially since inv_projData_sptr is 1s then this is an inversion.
              // inv_projdata_sptr = 1/norm3d
              norm3d_sptr->undo(*inv_projdata_3d_sptr);

              info("ScatterEstimation: Performing SSRB on efficiency factors ...");

              norm_projdata_2d_sptr=make_2D_projdata_sptr(inv_projdata_3d_sptr);

              // Crucial: Avoid divisions by zero!!
              // This should be resolved after https://github.com/UCL/STIR/issues/348
              pow_times_add min_threshold (0.0f, 1.0f, 1.0f,  1E-20f, NumericInfo<float>().max_value());
              apply_to_proj_data(*norm_projdata_2d_sptr, min_threshold);

              pow_times_add invert (0.0f, 1.0f, -1.0f, NumericInfo<float>().min_value(), NumericInfo<float>().max_value());
              apply_to_proj_data(*norm_projdata_2d_sptr, invert);

              norm_coeff_2d_sptr.reset(new BinNormalisationFromProjData(norm_projdata_2d_sptr));
            }
          else
            {
              norm_coeff_2d_sptr = norm3d_sptr;
            }

          shared_ptr<BinNormalisationFromProjData>atten_coeff_2d_sptr(new BinNormalisationFromProjData(atten_projdata_2d_sptr));
          this->multiplicative_binnorm_2d_sptr.reset(
                                                     new ChainedBinNormalisation(norm_coeff_2d_sptr, atten_coeff_2d_sptr));

          this->multiplicative_binnorm_2d_sptr->set_up(this->back_projdata_sptr->get_exam_info_sptr(), this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
          iterative_object->get_objective_function_sptr()->set_normalisation_sptr(multiplicative_binnorm_2d_sptr);
        }
      else // run_in_2d_projdata
        iterative_object->get_objective_function_sptr()->set_normalisation_sptr(multiplicative_binnorm_sptr);
    }
    info("ScatterEstimation: Done normalisation coefficients.");

    //
    // Set background (randoms) projdata
    //
    info("ScatterEstimation: 5.Calculating the background data and data_to_fit for the scaling...");

    if (!is_null_ptr(this->back_projdata_sptr))
    {
        if( back_projdata_sptr->get_num_segments() > 1)
        {
            info("ScatterEstimation: Running SSRB on the background data ...");


            this->back_projdata_2d_sptr=make_2D_projdata_sptr(back_projdata_sptr);
        }
        else
        {
            this->back_projdata_2d_sptr = back_projdata_sptr;
        }
    }
    else // We will need a background for the scatter, so let's create a simple empty ProjData
    {
        if (run_in_2d_projdata)
        {
            std::string out_filename = "tmp_background_data_2d.hs";

            this->back_projdata_2d_sptr = create_new_proj_data(out_filename,
                                                               this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                               this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
            this->back_projdata_2d_sptr->fill(0.0f);
        }
        else
        {
            std::string out_filename = "tmp_background_data.hs";

            this->back_projdata_sptr = create_new_proj_data(out_filename,
                                                            this->input_projdata_sptr->get_exam_info_sptr(),
                                                            this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());
            this->back_projdata_sptr->fill(0.0f);
        }
    }



    if (run_in_2d_projdata)
    {
        // Normalise in order to get the additive component
        std::stringstream convert;   // stream used for the conversion
        convert << output_additive_estimate_prefix << "_0_2d.hs";

        std::string out_filename = convert.str(); //extras_path.get_path() +"/"+ output_background_estimate_prefix + "";
        add_projdata_2d_sptr = create_new_proj_data(out_filename,
                                                    this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                    this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
        add_projdata_2d_sptr->fill(*back_projdata_2d_sptr);
        this->multiplicative_binnorm_2d_sptr->apply(*this->add_projdata_2d_sptr);

        iterative_object->get_objective_function_sptr()->set_additive_proj_data_sptr(this->add_projdata_2d_sptr);

        out_filename ="data_to_fit_2d.hs";
        data_to_fit_projdata_sptr = create_new_proj_data(out_filename,
                                                         this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                         this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());

        data_to_fit_projdata_sptr->fill(*input_projdata_2d_sptr);
        subtract_proj_data(*data_to_fit_projdata_sptr, *this->back_projdata_2d_sptr);
    }
    else
    {
        // Normalise in order to get the additive component
        std::string out_filename = output_additive_estimate_prefix + "_0.hs";
        add_projdata_sptr = create_new_proj_data(out_filename,
                                                 this->input_projdata_sptr->get_exam_info_sptr(),
                                                 this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());
        add_projdata_sptr->fill(*back_projdata_sptr);
        this->multiplicative_binnorm_sptr->apply(*this->add_projdata_sptr);

        iterative_object->get_objective_function_sptr()->set_additive_proj_data_sptr(this->add_projdata_sptr);

        out_filename = "data_to_fit.hs";
        data_to_fit_projdata_sptr = create_new_proj_data(out_filename,
                                                         this->input_projdata_sptr->get_exam_info_sptr(),
                                                         this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());
        data_to_fit_projdata_sptr->fill(*input_projdata_sptr);
        subtract_proj_data(*data_to_fit_projdata_sptr, *this->back_projdata_sptr);
    }

    return Succeeded::yes;
}

Succeeded
ScatterEstimation::
set_up_analytic()
{
    //TODO : I have most stuff in tmp.
     error("Analytic recon not implemented yet");
    return Succeeded::yes;
}

Succeeded
ScatterEstimation::
process_data()
{

  if (!this->_already_setup)
    error("ScatterEstimation: set_up needs to be called before process_data()");

    float local_min_scale_value = 0.5f;
    float local_max_scale_value = 0.5f;

    stir::BSpline::BSplineType  spline_type = stir::BSpline::quadratic;

    // This has been set to 2D or 3D in the set_up()
    shared_ptr <ProjData> unscaled_est_projdata_sptr(new ProjDataInMemory(this->scatter_simulation_sptr->get_exam_info_sptr(),
                                                                          this->scatter_simulation_sptr->get_template_proj_data_info_sptr()->create_shared_clone()));
    scatter_simulation_sptr->set_output_proj_data_sptr(unscaled_est_projdata_sptr);

    // Here the scaled scatter data will be stored.
    // Wether 2D or 3D depends on how the ScatterSimulation was initialised
    shared_ptr<ProjData> scaled_est_projdata_sptr;

    shared_ptr<BinNormalisation> normalisation_factors_sptr =
        this->get_normalisation_object_sptr(run_in_2d_projdata
					    ? this->multiplicative_binnorm_2d_sptr
					    : this->multiplicative_binnorm_sptr);
    if(run_in_2d_projdata)
    {
        scaled_est_projdata_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                            this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone()));
        scaled_est_projdata_sptr->fill(0.F);
    }
    else
    {
        scaled_est_projdata_sptr.reset(new ProjDataInMemory(this->input_projdata_sptr->get_exam_info_sptr(),
                                                            this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone()));
        scaled_est_projdata_sptr->fill(0.F);
    }

    info("ScatterEstimation: Start processing...");
    shared_ptr<DiscretisedDensity <3,float> > act_image_for_averaging;

    //Recompute the initial y image if the max is equal to the min.
#if 1
    if( this->current_activity_image_sptr->find_max() == this->current_activity_image_sptr->find_min() )
    {
        info("ScatterEstimation: The max and the min values of the current activity image are equal."
             "We deduce that it has been initialised to some value, therefore we will run an initial "
             "reconstruction ...");

        if (iterative_method)
            reconstruct_iterative(0, this->current_activity_image_sptr);
        else
            reconstruct_analytic(0, this->current_activity_image_sptr);

        if ( run_debug_mode )
        {
            std::string out_filename = extras_path.get_path() + "initial_activity_image";
            OutputFileFormat<DiscretisedDensity < 3, float > >::default_sptr()->
                    write_to_file(out_filename, *this->current_activity_image_sptr);
        }
    }
#else
    {
        std::string filename = extras_path.get_path() + "recon_0.hv";
        current_activity_image_sptr = read_from_file<DiscretisedDensity<3,float> >(filename);
    }
#endif
    // Set the first activity image
    scatter_simulation_sptr->set_activity_image_sptr(current_activity_image_sptr);

    if (this->do_average_at_2)
        act_image_for_averaging.reset(this->current_activity_image_sptr->clone());

    //
    // Begin the estimation process...
    //
    info("ScatterEstimation: Begin the estimation process...");
    for (int i_scat_iter = 1;
         i_scat_iter <= this->num_scatter_iterations;
         i_scat_iter++)
    {


        if ( this->do_average_at_2)
        {
            if (i_scat_iter == 2) // do average 0 and 1
            {
                if (is_null_ptr(act_image_for_averaging))
                    error("Storing the first activity estimate has failed at some point.");

                *this->current_activity_image_sptr += *act_image_for_averaging;
                *this->current_activity_image_sptr /= 2.f;
            }
        }

        if (!iterative_method)
        {
            // Threshold
        }

        info("ScatterEstimation: Scatter simulation in progress...");
        if (this->scatter_simulation_sptr->set_up() == Succeeded::no)
            error("ScatterEstimation: Failure at set_up() of the Scatter Simulation.");

        if (this->scatter_simulation_sptr->process_data() == Succeeded::no)
            error("ScatterEstimation: Scatter simulation failed");
        info("ScatterEstimation: Scatter simulation done...");

        if(this->run_debug_mode) // Write unscaled scatter sinogram
        {
            std::stringstream convert;   // stream used for the conversion
            convert << "unscaled_" << i_scat_iter;
            FilePath tmp(convert.str(),false);
            tmp.prepend_directory_name(extras_path.get_path());
            unscaled_est_projdata_sptr->write_to_file(tmp.get_string());
        }

        // Set the min and max scale factors
        // We're going to assume that the first iteration starts from an image without scatter correction, and therefore
        // overestimates scatter. This could be inaccurate, but is the case most of the time.
        // TODO introduce a variable to control this behaviour
        if (i_scat_iter > 0)
        {
            local_max_scale_value = this->max_scale_value;
            local_min_scale_value = this->min_scale_value;
        }

        scaled_est_projdata_sptr->fill(0.F);

        upsample_and_fit_scatter_estimate(*scaled_est_projdata_sptr, *data_to_fit_projdata_sptr,
                                          *unscaled_est_projdata_sptr,
                                          *normalisation_factors_sptr,
                                          *this->mask_projdata_sptr, local_min_scale_value,
                                          local_max_scale_value, this->half_filter_width,
                                          spline_type, true);

        if(this->run_debug_mode)
        {
            std::stringstream convert;   // stream used for the conversion
            convert << "scaled_" << i_scat_iter;
            FilePath tmp(convert.str(),false);
            tmp.prepend_directory_name(extras_path.get_path());
            scaled_est_projdata_sptr->write_to_file(tmp.get_string());
        }


        // When saving we need to go 3D.
        if (this->export_scatter_estimates_of_each_iteration ||
                i_scat_iter == this->num_scatter_iterations )
        {

            shared_ptr <ProjData> temp_scatter_projdata;

            if(run_in_2d_projdata)
            {
	        info("ScatterEstimation: upsampling scatter to 3D");
                //this is complicated as the 2d scatter estimate was
                //"unnormalised" (divided by norm2d), so we need to undo this 2D norm, and put a 3D norm in.
                //unfortunately, currently the values in the gaps in the
                //scatter estimate are not quite zero (just very small)
                //so we have to first make sure that they are zero before
                //we do any of this, otherwise the values after normalisation will be garbage
                //we do this by min-thresholding and then subtracting the threshold.
                //as long as the threshold is tiny, this will be ok

                // At the same time we are going to save to a temp projdata file

                shared_ptr<ProjData> temp_projdata ( new ProjDataInMemory (scaled_est_projdata_sptr->get_exam_info_sptr(),
                                                                           scaled_est_projdata_sptr->get_proj_data_info_sptr()));
                temp_projdata->fill(*scaled_est_projdata_sptr);
                pow_times_add min_threshold (0.0f, 1.0f, 1.0f, 1e-9f, NumericInfo<float>().max_value());
                pow_times_add add_scalar (-1e-9f, 1.0f, 1.0f, NumericInfo<float>().min_value(), NumericInfo<float>().max_value());
                apply_to_proj_data(*temp_projdata, min_threshold);
                apply_to_proj_data(*temp_projdata, add_scalar);
		// threshold back to 0 to avoid getting tiny negatives (due to numerical precision errors)
                pow_times_add min_threshold_zero (0.0f, 1.0f, 1.0f, 0.f, NumericInfo<float>().max_value());
                apply_to_proj_data(*temp_projdata, min_threshold_zero);

                // ok, we can multiply with the norm
                normalisation_factors_sptr->apply(*temp_projdata);

		// Create proj_data to save the 3d scatter estimate
                if(!this->output_scatter_estimate_prefix.empty())
                {
                    std::stringstream convert;
                    convert << this->output_scatter_estimate_prefix << "_" << i_scat_iter;
                    std::string output_scatter_filename = convert.str();

                    scatter_estimate_sptr.reset(
                                new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                      this->input_projdata_sptr->get_proj_data_info_sptr() ,
                                                      output_scatter_filename,
                                                      std::ios::in | std::ios::out | std::ios::trunc));
                }
                else
                {
		    // TODO should check if we have one already from previous iteration
                    scatter_estimate_sptr.reset(
                                new ProjDataInMemory(this->input_projdata_sptr->get_exam_info_sptr(),
                                                      this->input_projdata_sptr->get_proj_data_info_sptr()));
                }
		scatter_estimate_sptr->fill(0.0);

                // Upsample to 3D
                //we're currently not doing the tail fitting in this step, but keeping the same scale as determined in 2D
		//Note that most of the arguments here are ignored because we fix the scale to 1
		shared_ptr<BinNormalisation> normalisation_factors_3d_sptr =
		  this->get_normalisation_object_sptr(this->multiplicative_binnorm_sptr);

                upsample_and_fit_scatter_estimate(*scatter_estimate_sptr,
                                                  *this->input_projdata_sptr,
                                                  *temp_projdata,
                                                  *normalisation_factors_3d_sptr,
                                                  *this->input_projdata_sptr,
                                                  1.0f, 1.0f, 1, spline_type,
                                                  false);
	    }
	    else
	    {
	        scatter_estimate_sptr = scaled_est_projdata_sptr;
	    }

	    if(!this->output_additive_estimate_prefix.empty())
	    {
		info("ScatterEstimation: constructing additive sinogram");
		// Now save the full background term.
		std::stringstream convert;
		convert << this->output_additive_estimate_prefix << "_" <<
		  i_scat_iter;
		std::string output_additive_filename = convert.str();

		shared_ptr<ProjData> temp_additive_projdata(
					     new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
								   this->input_projdata_sptr->get_proj_data_info_sptr() ,
								   output_additive_filename,
								   std::ios::in | std::ios::out | std::ios::trunc));

		temp_additive_projdata->fill(*scatter_estimate_sptr);
		if (!is_null_ptr(this->back_projdata_sptr))
		  {
		    add_proj_data(*temp_additive_projdata, *this->back_projdata_sptr);
		  }

		this->multiplicative_binnorm_sptr->apply(*temp_additive_projdata);
	    }
        }

        // In the additive put the scaled scatter estimate
        // If we have randoms, then add them to the scaled scatter estimate
        // Then normalise
        if(run_in_2d_projdata)
        {
            this->add_projdata_2d_sptr->fill(*scaled_est_projdata_sptr);

            if (!is_null_ptr(this->back_projdata_2d_sptr))
            {
                add_proj_data(*add_projdata_2d_sptr, *this->back_projdata_2d_sptr);
            }
            this->multiplicative_binnorm_2d_sptr->apply(*add_projdata_2d_sptr);
        }
        else
        {
	    // TODO restructure code to move additive_projdata code from above
            error("ScatterEstimation: You should not be here. This is not 2D.");
        }
        current_activity_image_sptr->fill(1.f);

        iterative_method ? reconstruct_iterative(i_scat_iter,
                                                 this->current_activity_image_sptr):
                           reconstruct_analytic(i_scat_iter, this->current_activity_image_sptr);

        scatter_simulation_sptr->set_activity_image_sptr(current_activity_image_sptr);

    }

    info("ScatterEstimation: Scatter Estimation finished !!!");

    return Succeeded::yes;
}

void
ScatterEstimation::
reconstruct_iterative(int _current_iter_num,
                      shared_ptr<DiscretisedDensity<3, float> > & _current_estimate_sptr)
{

    shared_ptr<IterativeReconstruction<DiscretisedDensity<3, float> > > tmp_iterative =
            dynamic_pointer_cast<IterativeReconstruction<DiscretisedDensity<3, float> > >(reconstruction_template_sptr);

    //
    // Now, we can call Reconstruction::set_up().
    if (tmp_iterative->set_up(this->current_activity_image_sptr) == Succeeded::no)
    {
        error("ScatterEstimation: Failure at set_up() of the reconstruction method. Aborting.");
    }

    //    return iterative_object->reconstruct(this->activity_image_lowres_sptr);
    tmp_iterative->reconstruct(this->current_activity_image_sptr);

    if(this->run_debug_mode)
    {
        std::stringstream convert;   // stream used for the conversion
        convert << "recon_" << _current_iter_num;
        FilePath tmp(convert.str(),false);
        tmp.prepend_directory_name(extras_path.get_path());
        OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                write_to_file(tmp.get_string(), *_current_estimate_sptr);

    }
}

void
ScatterEstimation::
reconstruct_analytic(int _current_iter_num,
                     shared_ptr<DiscretisedDensity<3, float> > & _current_estimate_sptr)
{
    AnalyticReconstruction* analytic_object =
            dynamic_cast<AnalyticReconstruction* > (this->reconstruction_template_sptr.get());
    analytic_object->reconstruct(this->current_activity_image_sptr);

    if(this->run_debug_mode)
    {
        std::stringstream convert;   // stream used for the conversion
        convert << "recon_analytic_"<< _current_iter_num;
        FilePath tmp(convert.str(),false);
        tmp.prepend_directory_name(extras_path.get_path());
        OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                write_to_file(tmp.get_string(), *_current_estimate_sptr);

    }

    //TODO: threshold ... to cut the negative values
}

/****************** functions to help **********************/

void
ScatterEstimation::
add_proj_data(ProjData& first_addend, const ProjData& second_addend)
{
    assert(first_addend.get_min_segment_num() == second_addend.get_min_segment_num());
    assert(first_addend.get_max_segment_num() == second_addend.get_max_segment_num());
    for (int segment_num = first_addend.get_min_segment_num();
         segment_num <= first_addend.get_max_segment_num();
         ++segment_num)
    {
        SegmentByView<float> first_segment_by_view =
                first_addend.get_segment_by_view(segment_num);

        SegmentByView<float> sec_segment_by_view =
                second_addend.get_segment_by_view(segment_num);

        first_segment_by_view += sec_segment_by_view;

        if (!(first_addend.set_segment(first_segment_by_view) == Succeeded::yes))
        {
            error("Error set_segment %d", segment_num);
        }
    }
}

void
ScatterEstimation::
subtract_proj_data(ProjData& minuend, const ProjData& subtracted)
{
    assert(minuend.get_min_segment_num() == subtracted.get_min_segment_num());
    assert(minuend.get_max_segment_num() == subtracted.get_max_segment_num());
    for (int segment_num = minuend.get_min_segment_num();
         segment_num <= minuend.get_max_segment_num();
         ++segment_num)
    {
        SegmentByView<float> first_segment_by_view =
                minuend.get_segment_by_view(segment_num);

        SegmentByView<float> sec_segment_by_view =
                subtracted.get_segment_by_view(segment_num);

        first_segment_by_view -= sec_segment_by_view;

        if (!(minuend.set_segment(first_segment_by_view) == Succeeded::yes))
        {
            error("ScatterEstimation: Error set_segment %d", segment_num);
        }
    }

    // Filter negative values:
    //    pow_times_add zero_threshold (0.0f, 1.0f, 1.0f, 0.0f, NumericInfo<float>().max_value());
    //    apply_to_proj_data(minuend, zero_threshold);
}

void
ScatterEstimation::
apply_to_proj_data(ProjData& data, const pow_times_add& func)
{
    for (int segment_num = data.get_min_segment_num();
         segment_num <= data.get_max_segment_num();
         ++segment_num)
    {
        SegmentByView<float> segment_by_view =
                data.get_segment_by_view(segment_num);

        in_place_apply_function(segment_by_view,
                                func);

        if (!(data.set_segment(segment_by_view) == Succeeded::yes))
        {
            error("ScatterEstimation: Error set_segment %d", segment_num);
        }
    }
}

Succeeded
ScatterEstimation::project_mask_image()
{
    if (is_null_ptr(this->mask_image_sptr))
    {
        warning("You cannot forward project if you have not set the mask image. Aborting.");
        return Succeeded::no;
    }

    if (run_in_2d_projdata)
    {
        if (is_null_ptr(this->input_projdata_2d_sptr))
        {
            warning("No 2D proj_data have been initialised. Aborting.");
            return Succeeded::no;
        }
    }
    else
    {
        if (is_null_ptr(this->input_projdata_sptr))
        {
            warning("No 3D proj_data have been initialised. Aborting.");
            return Succeeded::no;
        }
    }

    shared_ptr<ForwardProjectorByBin> forw_projector_sptr;
    shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
    forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    info(boost::format("ScatterEstimation: Forward projector used for the calculation of "
                       "the tail mask: %1%") % forw_projector_sptr->parameter_info());

    shared_ptr<ProjData> mask_projdata;
    if(run_in_2d_projdata)
    {
        forw_projector_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_sptr(),
                                    this->mask_image_sptr );

        mask_projdata.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                 this->input_projdata_2d_sptr->get_proj_data_info_sptr()));
    }
    else
    {
        forw_projector_sptr->set_up(this->input_projdata_sptr->get_proj_data_info_sptr(),
                                    this->mask_image_sptr );

        mask_projdata.reset(new ProjDataInMemory(this->input_projdata_sptr->get_exam_info_sptr(),
                                                 this->input_projdata_sptr->get_proj_data_info_sptr()));
    }

    forw_projector_sptr->forward_project(*mask_projdata, *this->mask_image_sptr);

    //add 1 to be able to use create_tail_mask_from_ACFs (which expects ACFs,
    //so complains if the threshold is too low)

    pow_times_add pow_times_add_object(1.0f, 1.0f, 1.0f,NumericInfo<float>().min_value(),
                                       NumericInfo<float>().max_value());

    // I have only one segment I could remove this.
    for (int segment_num = mask_projdata->get_min_segment_num();
         segment_num <= mask_projdata->get_max_segment_num();
         ++segment_num)
    {
        SegmentByView<float> segment_by_view =
                mask_projdata->get_segment_by_view(segment_num);

        in_place_apply_function(segment_by_view,
                                pow_times_add_object);

        if (!(mask_projdata->set_segment(segment_by_view) == Succeeded::yes))
        {
            warning("ScatterEstimation: Error set_segment %d", segment_num);
            return Succeeded::no;
        }
    }

    if (this->mask_projdata_filename.size() > 0)
        this->mask_projdata_sptr.reset(new ProjDataInterfile(mask_projdata->get_exam_info_sptr(),
                                                             mask_projdata->get_proj_data_info_sptr(),
                                                             this->mask_projdata_filename,
                                                             std::ios::in | std::ios::out | std::ios::trunc));
    else
        this->mask_projdata_sptr.reset(new ProjDataInMemory(mask_projdata->get_exam_info_sptr(),
                                                            mask_projdata->get_proj_data_info_sptr()));

    CreateTailMaskFromACFs create_tail_mask_from_acfs;

    if(this->tail_mask_par_filename.empty())
    {
        create_tail_mask_from_acfs.ACF_threshold = 1.1;
        create_tail_mask_from_acfs.safety_margin = 4;
    }
    else
    {
        if(!create_tail_mask_from_acfs.parse(this->tail_mask_par_filename.c_str()))
            error(boost::format("Error parsing parameters file %1%, for creating mask tails from ACFs.")
                 %this->tail_mask_par_filename);
    }
    
    create_tail_mask_from_acfs.set_input_projdata_sptr(mask_projdata);
    create_tail_mask_from_acfs.set_output_projdata_sptr(this->mask_projdata_sptr);
    return create_tail_mask_from_acfs.process_data();
}

void
ScatterEstimation::
apply_mask_in_place(DiscretisedDensity<3, float>& arg,
                    const MaskingParameters& masking_parameters)
{
  if (!is_null_ptr(masking_parameters.filter_sptr))
    {      
      masking_parameters.filter_sptr->process_data(arg);
    }

  // min threshold
  for (DiscretisedDensity<3,float>::full_iterator iter = arg.begin_all(); iter != arg.end_all(); ++iter)
    {
      if (*iter < masking_parameters.min_threshold)
	*iter = 0.F;
      else
	*iter = 1.F;
    }
}

int ScatterEstimation::get_num_iterations() const
{
    return num_scatter_iterations;
}

// deprecated version
int ScatterEstimation::get_iterations_num() const
{
    return num_scatter_iterations;
}

void
ScatterEstimation::create_multiplicative_binnorm_sptr()
{
  if (!is_null_ptr(this->multiplicative_binnorm_sptr))
    {
      if (!is_null_ptr(this->norm_3d_sptr))
        error("ScatterEstimation: cannot handle having both norm and 'combined norm' initialised");
      if (!is_null_ptr(this->atten_norm_3d_sptr))
        error("ScatterEstimation: cannot handle having both attenuation and 'combined norm' initialised");
    }
  else
    {
      if (is_null_ptr(this->atten_norm_3d_sptr))
        {
          error("ScatterEstimation: need attenuation correction factors to be set (sorry)");
        }
      if (is_null_ptr(this->norm_3d_sptr))
        {
          warning("ScatterEstimation: no normalisation data set. This would only be appropriate for simple simulations.");
          this->norm_3d_sptr = this->atten_norm_3d_sptr;
        }
      else
        {
          this->multiplicative_binnorm_sptr.reset(new ChainedBinNormalisation(norm_3d_sptr, atten_norm_3d_sptr));
        }
    }
}

shared_ptr<BinNormalisation>
ScatterEstimation::get_normalisation_object_sptr(const shared_ptr<BinNormalisation>& combined_norm_sptr) const
{
    const ChainedBinNormalisation* tmp_chain_norm_sptr =
      dynamic_cast<const ChainedBinNormalisation*>(combined_norm_sptr.get());

    if (!is_null_ptr(tmp_chain_norm_sptr ))
    {
        return tmp_chain_norm_sptr->get_first_norm();
    }
    else //Just trivial, then ..
    {
        shared_ptr<BinNormalisation> normalisation_factors_sptr(new TrivialBinNormalisation());
        normalisation_factors_sptr->set_up(this->input_projdata_sptr->get_exam_info_sptr(), this->input_projdata_sptr->get_proj_data_info_sptr());
	return normalisation_factors_sptr;
    }
}

shared_ptr<ProjData>
ScatterEstimation::get_attenuation_correction_factors_sptr(const shared_ptr<BinNormalisation>& combined_norm_sptr) const
{
  const ChainedBinNormalisation* tmp_chain_norm_sptr =
    dynamic_cast<const ChainedBinNormalisation*>(combined_norm_sptr.get());
  shared_ptr<BinNormalisation> atten_norm_sptr;
  if (!is_null_ptr(tmp_chain_norm_sptr ))
    {
      atten_norm_sptr = tmp_chain_norm_sptr->get_second_norm();
    }
  else
    {
      atten_norm_sptr = combined_norm_sptr;
    }

  return
    dynamic_cast<BinNormalisationFromProjData*> (atten_norm_sptr.get())->get_norm_proj_data_sptr();

}

shared_ptr<ProjData> ScatterEstimation::create_new_proj_data(const std::string& filename,
                                                             const shared_ptr<const ExamInfo> exam_info_sptr,
                                                             const shared_ptr<const ProjDataInfo> proj_data_info_sptr) const
{
    shared_ptr<ProjData> pd_sptr;
    if (run_debug_mode)
    {
        FilePath tmp(filename, false);
	tmp = tmp.get_filename(); // get rid of any folder info
	tmp.prepend_directory_name(extras_path.get_path());	
        pd_sptr.reset(new ProjDataInterfile(exam_info_sptr,
                                            proj_data_info_sptr,
                                            tmp.get_string(),
                                            std::ios::in | std::ios::out | std::ios::trunc));
    }
    else
    {
        pd_sptr.reset(new ProjDataInMemory(exam_info_sptr, proj_data_info_sptr));
    }
    return pd_sptr;
}

END_NAMESPACE_STIR
