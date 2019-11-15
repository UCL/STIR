/*
  Copyright (C) 2004 -  2009 Hammersmith Imanet Ltd
  Copyright (C) 2013,2016,2019 University College London
  Copyright (C) 2018-2019, University of Hull
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
  \brief Implementation of most functions in stir::ScatterEstimation

  \author Nikos Efthimiou
  \author Kris Thielemans
*/
#include "stir/scatter/ScatterEstimation.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/SSRB.h"
#include "stir/DataProcessor.h"
#include "stir/scatter/CreateTailMaskFromACFs.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/zoom.h"
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

#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"

START_NAMESPACE_STIR

#define SPEED_UP_FOR_DEBUG 0

void
ScatterEstimation::
set_defaults()
{
    this->recompute_atten_projdata = true;
    this->recompute_mask_image = true;
    this->recompute_mask_projdata = true;
    this->iterative_method = true;
    this->do_average_at_2 = true;
    this->export_scatter_estimates_of_each_iteration = false;
    this->run_debug_mode = false;
    this->override_initial_activity_image = false;
    this->override_density_image = false;
    this->override_density_image_for_scatter_points = false;
    this->remove_interleaving = true;
    this->atten_image_filename = "";
    this->norm_coeff_filename = "";
    this->output_scatter_estimate_prefix = "";
    this->output_background_estimate_prefix = "";
    this->num_scatter_iterations = 5;
    this->min_scale_value = 0.4f;
    this->max_scale_value = 100.f;
    this->half_filter_width = 3;

    this->atten_coeff_sptr.reset(new TrivialBinNormalisation());
    normalisation_coeffs_sptr.reset(new TrivialBinNormalisation());
}

void
ScatterEstimation::
initialise_keymap()
{
    this->parser.add_start_key("Scatter Estimation Parameters");
    this->parser.add_stop_key("end Scatter Estimation Parameters");
    // N.E. 13/07/16: I don't like "input file" for the input data.
    // I try to keep consistency with the reconstruction
    // params.

    this->parser.add_key("run in debug mode",
                         &this->run_debug_mode);
    this->parser.add_key("input file",
                         &this->input_projdata_filename);
    this->parser.add_key("attenuation image filename",
                         &this->atten_image_filename);

    // MASK parameters
    this->parser.add_key("mask attenuation image filename",
                         &this->mask_image_filename);
    this->parser.add_key("mask image postfilter filename",
                         &this->mask_postfilter_filename);
    this->parser.add_key("recompute mask image",
                         &this->recompute_mask_image);
    this->parser.add_key("mask image max threshold ",
                         &this->mask_image.max_threshold);
    this->parser.add_key("mask image additive scalar",
                         &this->mask_image.add_scalar);
    this->parser.add_key("mask image min threshold",
                         &this->mask_image.min_threshold);
    this->parser.add_key("mask image times scalar",
                         &this->mask_image.times_scalar);
    this->parser.add_key("recompute mask projdata",
                         &this->recompute_mask_projdata);
    this->parser.add_key("mask projdata filename",
                         &this->mask_projdata_filename);
    this->parser.add_key("tail fitting par filename",
                         &this->tail_mask_par_filename);
    // END MASK

    this->parser.add_key("attenuation projdata filename",
                         &this->atten_coeff_filename);
    this->parser.add_key("recompute attenuation projdata",
                         &this->recompute_atten_projdata);
    this->parser.add_key("background projdata filename",
                         &this->back_projdata_filename);
    this->parser.add_parsing_key("Bin Normalisation type",
                         &this->normalisation_coeffs_sptr);

    // RECONSTRUCTION RELATED
    this->parser.add_key("reconstruction parameter template file",
                         &this->recon_template_par_filename);
    this->parser.add_parsing_key("reconstruction method",
                                 &this->reconstruction_template_sptr);
    // END RECONSTRUCTION RELATED

    this->parser.add_key("number of scatter iterations",
                         &this->num_scatter_iterations);
    //Scatter simulation
    this->parser.add_parsing_key("Simulation method",
                                 &this->scatter_simulation_sptr);
    this->parser.add_key("scatter simulation parameters file",
                         &this->scatter_sim_par_filename);
    this->parser.add_key("use default downsampling in scatter simulation",
                         &this->use_default_downsampling);

    this->parser.add_key("override initial activity image",
                         &this->override_initial_activity_image);
    this->parser.add_key("override density image",
                         &this->override_density_image);
    this->parser.add_key("override density image for scatter points",
                         &this->override_density_image_for_scatter_points);
    this->parser.add_key("override scanner template",
                         &this->override_scanner_template);

    // END Scatter simulation

    this->parser.add_key("export scatter estimates of each iteration",
                         &this->export_scatter_estimates_of_each_iteration);
    this->parser.add_key("output scatter estimate name prefix",
                         &this->output_scatter_estimate_prefix);
    this->parser.add_key("output background estimate name prefix",
                         &this->output_background_estimate_prefix);
    this->parser.add_key("do average at 2",
                         &this->do_average_at_2);
    this->parser.add_key("maximum scale value",
                         &this->max_scale_value);
    this->parser.add_key("minimum scale value",
                         &this->min_scale_value);
    this->parser.add_key("half filter width",
                         &this->half_filter_width);
    this->parser.add_key("remove interleaving",
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
    if (parameter_filename.size() == 0)
    {
        this->set_defaults();
        this->ask_parameters();
    }
    else
    {
        this->set_defaults();
        if (!this->parse(parameter_filename.c_str()))
        {
            error("ScatterEstimation: Error parsing input file %s. Aborting.", parameter_filename.c_str());
        }
    }
}

bool
ScatterEstimation::
post_processing()
{
    // Check that the crusial parts have been set.
    info("ScatterEstimation: Loading input projection data");
    if (this->input_projdata_filename.size() == 0)
    {
        warning("ScatterEstimation: No input projdata filename is given. Aborting.");
        return true;
    }

    this->input_projdata_sptr =
            ProjData::read_from_file(this->input_projdata_filename);

    // If the reconstruction_template_sptr is null then, we need to parse it from another
    // file. I prefer this implementation since makes smaller modular files.
    if (this->recon_template_par_filename.size() == 0)
    {
        warning("ScatterEstimation: Please define a reconstruction method. Aborting.");
        return true;
    }
    else
    {
        KeyParser local_parser;
        local_parser.add_start_key("Reconstruction");
        local_parser.add_stop_key("End Reconstruction");
        local_parser.add_parsing_key("reconstruction method", &this->reconstruction_template_sptr);
        if (!local_parser.parse(this->recon_template_par_filename.c_str()))
        {
            warning(boost::format("ScatterEstimation: Error parsing reconstruction parameters file %1%. Aborting.")
                    %this->recon_template_par_filename);
            return true;
        }
    }

    info("ScatterEstimation: Loading attenuation image...");
    if (this->atten_image_filename.size() == 0)
    {
        warning("ScatterEstimation: Please define an attenuation image. Aborting.");
        return true;
    }
    else
        this->atten_image_sptr =
            read_from_file<DiscretisedDensity<3,float> >(this->atten_image_filename);

    if(this->atten_coeff_filename.size() > 0 && !recompute_atten_projdata)
    {
        info("ScatterEstimation: Loading attenuation correction coefficients...");
        this->atten_coeff_sptr.reset(new BinNormalisationFromProjData(this->atten_coeff_filename));
    }

    if(is_null_ptr(normalisation_coeffs_sptr))
        warning("ScatterEstimation: No normalisation coefficients have been set!!");

    this->multiplicative_binnorm_sptr.reset(new ChainedBinNormalisation(normalisation_coeffs_sptr, atten_coeff_sptr));
    this->multiplicative_binnorm_sptr->set_up(this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());


    if (this->back_projdata_filename.size() > 0)
    {
        info("ScatterEstimation: Loading background projdata...");
        this->add_projdata_sptr =
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

    info ("ScatterEstimation: Initialising mask image ... ");
    if(this->mask_postfilter_filename.size() > 0 )
    {
        this->filter_sptr.reset(new PostFiltering <DiscretisedDensity<3,float> >);

        if(!filter_sptr->parse(this->mask_postfilter_filename.c_str()))
        {
            warning(boost::format("ScatterEstimation: Error parsing post filter parameters file %1%. Aborting.")
                    %this->mask_postfilter_filename);
            return true;
        }
    }

    info ("ScatterEstimation: Initialising Scatter Simulation ... ");
    if (this->scatter_sim_par_filename.size() == 0)
    {
        warning("ScatterEstimation: Please define a scatter simulation method. Aborting.");
        return true;
    }
    else // Parse locally
    {
        KeyParser local_parser;
        local_parser.add_start_key("Scatter Simulation");
        local_parser.add_stop_key("End Scatter Simulation");
        local_parser.add_parsing_key("Simulation method", &this->scatter_simulation_sptr);
        if (!local_parser.parse(this->scatter_sim_par_filename.c_str()))
        {
            warning(boost::format("ScatterEstimation: Error parsing scatter simulation parameters file %1%. Aborting.")
                    %this->recon_template_par_filename);
            return true;
        }
    }

    if (this->output_scatter_estimate_prefix.size() == 0)
        return true;

    if (this->output_background_estimate_prefix.size() == 0)
        return true;

    if(!this->recompute_mask_projdata)
    {
        if (this->mask_projdata_filename.size() == 0)
        {
            warning("ScatterEstimation: Please define a filename for mask proj_data. Aborting.");
            return true;
        }
        this->mask_projdata_sptr =
                ProjData::read_from_file(this->mask_projdata_filename);
    }
    else
    {
        if (!this->recompute_mask_image)
        {
            if (this->mask_image_filename.size() == 0 )
            {
                warning("ScatterEstimation: Please define a filename for mask image. Aborting.");
                return true;
            }

            this->mask_image_sptr =
                    read_from_file<DiscretisedDensity<3, float> >(this->mask_image_filename);
        }

        if (this->tail_mask_par_filename.size() == 0)
        {
            warning("ScatterEstimation: Please define a filename for tails mask. Aborting.");
            return true;
        }
    }

    return false;
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

    if (is_null_ptr(this->input_projdata_sptr))
    {
        warning("ScatterEstimation: No input proj_data have been set. Aborting.");
        return Succeeded::no;
    }

    // Calculate the SSRB
    if (input_projdata_sptr->get_num_segments() > 1)
    {
        info("ScatterEstimation: Running SSRB on input data...");
        shared_ptr<ProjDataInfo> proj_data_info_2d_sptr(
                    dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >
                    (SSRB(*this->input_projdata_sptr->get_proj_data_info_ptr(),
                          this->input_projdata_sptr->get_num_segments(), 1, false)));

        FilePath tmp(this->input_projdata_filename);
        std::string out_filename = extras_path.get_path() + tmp.get_filename_no_extension() + "_2d.hs";

        this->input_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                 proj_data_info_2d_sptr,
                                                                 out_filename,
                                                                 std::ios::in | std::ios::out | std::ios::trunc));
        SSRB(*this->input_projdata_2d_sptr,
             *input_projdata_sptr,false);
    }
    else
    {
        input_projdata_2d_sptr = input_projdata_sptr;
    }

info("ScatterEstimation: Setting up reconstruction method ...");

if(is_null_ptr(this->reconstruction_template_sptr))
{
    warning("ScatterEstimation: Reconstruction method has not been initialised. Aborting.");
    return Succeeded::no;
}

    // We have to check which reconstruction method we are going to use ...
    shared_ptr<AnalyticReconstruction> tmp_analytic =
            dynamic_pointer_cast<AnalyticReconstruction >(this->reconstruction_template_sptr);
    shared_ptr<IterativeReconstruction<DiscretisedDensity<3, float> >> tmp_iterative =
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

    //    if ( run_debug_mode )
    //    {
    //        std::string out_filename = extras_path.get_path() + "inital_activity_image";
    //        OutputFileFormat<DiscretisedDensity < 3, float > >::default_sptr()->
    //                write_to_file(out_filename, *this->current_activity_image_sptr);
    //    }

    // Based on the activity image zoom the attenuation
    if (!is_null_ptr(current_activity_image_sptr))
    {
        VoxelsOnCartesianGrid<float>* tmp_act = dynamic_cast<VoxelsOnCartesianGrid<float>* >(current_activity_image_sptr.get());
        VoxelsOnCartesianGrid<float>* tmp_att = dynamic_cast<VoxelsOnCartesianGrid<float>* >(atten_image_sptr.get());

        float _zoom_xy =
                tmp_att->get_voxel_size().x() / tmp_act->get_voxel_size().x();
        float _zoom_z =
                tmp_att->get_voxel_size().z() / tmp_act->get_voxel_size().z();

        BasicCoordinate<3,int> new_size = make_coordinate(tmp_act->get_z_size(),
                                                          tmp_act->get_y_size(),
                                                          tmp_act->get_x_size());

        zoom_image_in_place(*tmp_att ,
                            CartesianCoordinate3D<float>(_zoom_z, _zoom_xy, _zoom_xy),
                            CartesianCoordinate3D<float>(0,0,0),
                            new_size);

        *tmp_att *= _zoom_xy * _zoom_xy * _zoom_z;
    }

    //    if ( run_debug_mode )
    //    {
    //        std::string out_filename = extras_path.get_path() + "mod_atten_image";
    //        OutputFileFormat<DiscretisedDensity < 3, float > >::default_sptr()->
    //                write_to_file(out_filename, *this->atten_image_sptr);
    //    }

    //        if ( run_debug_mode )
    //        {
    //            std::string out_filename = extras_path.get_path() + "inital_activity_image";
    //            OutputFileFormat<DiscretisedDensity < 3, float > >::default_sptr()->
    //                    write_to_file(out_filename, *this->current_activity_image_sptr);
    //        }


    //
    // ScatterSimulation
    //

    info("ScatterEstimation: Setting up Scatter Simulation method ...");
    if(is_null_ptr(this->scatter_simulation_sptr))
    {
        warning("Scatter simulation method has not been initialised. Aborting.");
        return Succeeded::no;
    }

    // The images are passed to the simulation.
    // and it will override anything that the ScatterSimulation.par file has done.
    if(this->override_density_image)
    {
        info("ScatterEstimation: Over-riding attenuation image! (The file and settings set in the simulation par file are discarded)");
        this->scatter_simulation_sptr->set_density_image_sptr(this->atten_image_sptr);
    }

    if(this->override_density_image_for_scatter_points)
    {
        info("ScatterEstimation: Over-riding attenuation image for scatter points! (The file and settings set in the simulation par file are discarded)");
        this->scatter_simulation_sptr->set_density_image_for_scatter_points_sptr(this->atten_image_sptr);
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
            this->scatter_simulation_sptr->set_template_proj_data_info_sptr(this->input_projdata_2d_sptr->get_proj_data_info_sptr());
        }
        else
        {
            this->scatter_simulation_sptr->set_template_proj_data_info_sptr(this->input_projdata_sptr->get_proj_data_info_sptr());
        }
        this->scatter_simulation_sptr->set_exam_info_sptr(this->input_projdata_2d_sptr->get_exam_info_sptr());
    }

    //    if (this->scatter_simulation_sptr->set_up() == Succeeded::no)
    //    {
    //        warning ("ScatterEstimation: Failure at set_up() of the Scatter Simulation. Aborting.");
    //        return Succeeded::no;
    //    }

    if (this->use_default_downsampling)
        this->scatter_simulation_sptr->default_downsampling(false);

    // Check if Load a mask proj_data

    if(is_null_ptr(this->mask_projdata_sptr) || this->recompute_mask_projdata)
    {
        if(is_null_ptr(this->mask_image_sptr) || this->recompute_mask_image)
        {
            // Applying mask
            // 1. Clone from the original image.
            // 2. Apply to the new clone.
            this->mask_image_sptr.reset(this->atten_image_sptr->clone());
            if(this->apply_mask_in_place(*this->mask_image_sptr,
                                         this->mask_image) == false)
            {
                warning("Error in masking. Aborting.");
                return Succeeded::no;
            }

            if (this->mask_image_filename.size() > 0 )
                OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                        write_to_file(this->mask_image_filename, *this->mask_image_sptr.get());
        }

        if(ffw_project_mask_image() == Succeeded::no)
        {
            warning("ScatterEstimation: Unsuccessfull to fwd project the mask image. Aborting.");
            return Succeeded::no;
        }
    }

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

    const double start_time = this->input_projdata_sptr->get_exam_info_sptr()->get_time_frame_definitions().get_start_time();
    const double end_time =this->input_projdata_sptr->get_exam_info_sptr()->get_time_frame_definitions().get_end_time();


    //    //
    //    // Multiplicative projdata
    //    //

//#if SPEED_UP_FOR_DEBUG == 0
//    // If second is trivial attenuation proj_data have not been set, yet

//    if (this->multiplicative_binnorm_sptr->is_second_trivial())
//    {

//        if(recompute_atten_projdata)// In that case use the attenuation image to create it
//        {
//            shared_ptr<BinNormalisation> tmp_attenuation_correction_sptr(new TrivialBinNormalisation());
//            shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
//            shared_ptr<ForwardProjectorByBin> forw_projector_sptr(new ForwardProjectorByBinUsingProjMatrixByBin(PM));

//            {
//                warning("ScatterEstimation: No attenuation projdata have been initialised."
//                        "BinNormalisationFromAttenuationImage will be used. It is slower in general.");

//                tmp_attenuation_correction_sptr.reset(new BinNormalisationFromAttenuationImage(this->atten_image_sptr,
//                                                                                               forw_projector_sptr));
//            }

//            {
//                shared_ptr<ProjData> atten_projdata_3d_sptr;

//                if (this->atten_coeff_filename.size() > 0 )
//                    atten_projdata_3d_sptr.reset(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
//                                                                       this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone(),
//                                                                       this->atten_coeff_filename,
//                                                                       std::ios::in | std::ios::out | std::ios::trunc));
//                else // Maybe throw an error??
//                    atten_projdata_3d_sptr.reset(new ProjDataInMemory(this->input_projdata_sptr->get_exam_info_sptr(),
//                                                                      this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone()));

//                atten_projdata_3d_sptr->fill(1.f);

//                if (tmp_attenuation_correction_sptr->set_up(atten_projdata_3d_sptr->get_proj_data_info_ptr()->create_shared_clone())
//                        != Succeeded::yes)
//                {

//                    return Succeeded::no;
//                }

//                shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(forw_projector_sptr->get_symmetries_used()->clone());
//                info("ScatterEstimation: Calculating the attenuation projection data...");
//                tmp_attenuation_correction_sptr->apply(*atten_projdata_3d_sptr, start_time, end_time, symmetries_sptr);
//                this->atten_coeff_3d_sptr.reset(new BinNormalisationFromProjData(this->atten_coeff_filename));
//                atten_coeff_3d_sptr->set_up(this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());

//                this->multiplicative_binnorm_3d_sptr.reset(new ChainedBinNormalisation(normalisation_coeffs_3d_sptr, atten_coeff_3d_sptr));
//                this->multiplicative_binnorm_3d_sptr->set_up(this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());
//            }
//        }
//        else { // Normally it should never get in here.

//            this->atten_coeff_sptr.reset(new BinNormalisationFromProjData(this->atten_coeff_filename));
//            atten_coeff_sptr->set_up(this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());
//            this->multiplicative_binnorm_sptr.reset(new ChainedBinNormalisation(normalisation_coeffs_sptr, atten_coeff_sptr));
//            this->multiplicative_binnorm_sptr->set_up(this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());
//        }
//    }

//#endif

    shared_ptr<ProjData> atten_projdata_2d_sptr;


#if SPEED_UP_FOR_DEBUG == 0
    info("ScatterEstimation: 3.Calculating the attenuation projection data...");
    shared_ptr<ProjData> tmp_atten_projdata_sptr =
            dynamic_cast<BinNormalisationFromProjData*> (this->multiplicative_binnorm_sptr->get_second_norm().get())->get_norm_proj_data_sptr();

    if( tmp_atten_projdata_sptr->get_num_segments() > 1)
    {
        info("ScatterEstimation: Running SSRB on attenuation correction coefficients ...");

            FilePath tmp(this->atten_coeff_filename);
            std::string out_filename = extras_path.get_path() + tmp.get_filename_no_extension() + "_2d.hs";

            atten_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                               this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                               out_filename,
                                                               std::ios::in | std::ios::out | std::ios::trunc));
        SSRB(*atten_projdata_2d_sptr,
             *tmp_atten_projdata_sptr, true);
    }
    else
    {
        // TODO: this needs more work. -- Setting directly 2D proj_data is buggy right now.
        atten_projdata_2d_sptr = tmp_atten_projdata_sptr;
    }
#else
    FilePath tmp(this->atten_coeff_filename);
    std::string in_filename = extras_path.get_path() + tmp.get_filename_no_extension() + "_2d.hs";

    atten_projdata_2d_sptr = ProjData::read_from_file(in_filename);

#endif

    atten_coeff_2d_sptr.reset(new BinNormalisationFromProjData(atten_projdata_2d_sptr));
    atten_coeff_2d_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
    //<- End of Attenuation projdata

    // Normalisation ProjData


    if(!this->multiplicative_binnorm_sptr->is_first_trivial())
    {
        // Check if 3D
        shared_ptr<ProjData> tmp_norm_projdata_sptr =
                dynamic_cast<BinNormalisationFromProjData*> (this->multiplicative_binnorm_sptr->get_first_norm().get())->get_norm_proj_data_sptr();

        if ( tmp_norm_projdata_sptr->get_num_segments() > 1) // This means that we have set a normalisation sinogram.
        {
            // N.E.: From K.T.: Bin normalisation doesn't know about SSRB.
            // we need to get norm2d=1/SSRB(1/norm3d))

            info("ScatterEstimation: Constructing 2D normalisation coefficients ...");

            std::string out_filename = extras_path.get_path() + "tmp_inverted_normdata.hs";
            shared_ptr<ProjData> inv_projdata_3d_sptr(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                            this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                            out_filename,
                                                                            std::ios::in | std::ios::out | std::ios::trunc));
            inv_projdata_3d_sptr->fill(1.f);

            out_filename = extras_path.get_path() + "tmp_projdata_2d.hs";
            shared_ptr<ProjData> tmp_projdata_2d_sptr(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                            this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                            out_filename,
                                                                            std::ios::in | std::ios::out | std::ios::trunc));
            tmp_projdata_2d_sptr->fill(1.f);

            out_filename = extras_path.get_path() + "tmp_normdata_2d.hs";
            shared_ptr<ProjData> norm_projdata_2d_sptr(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                             this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                             out_filename,
                                                                             std::ios::in | std::ios::out | std::ios::trunc));
            norm_projdata_2d_sptr->fill(1.f);

            // Essentially since inv_projData_sptr is 1s then this is an inversion.
            // inv_projdata_sptr = 1/norm3d
            this->multiplicative_binnorm_3d_sptr->undo_only_first(*inv_projdata_3d_sptr, start_time, end_time);

            info("ScatterEstimation: Performing SSRB on normalisation coefficients ...");

            SSRB(*tmp_projdata_2d_sptr,
                 *inv_projdata_3d_sptr,false);

            // Crucial: Avoid divisions by zero!!
            // This should be resolved after https://github.com/UCL/STIR/issues/348
            pow_times_add min_threshold (0.0f, 1.0f, 1.0f,  1E-20f, NumericInfo<float>().max_value());
//            pow_times_add add_scalar (1e-4f, 1.0f, 1.0f, NumericInfo<float>().min_value(), NumericInfo<float>().max_value());

            apply_to_proj_data(*tmp_projdata_2d_sptr, min_threshold);
//            apply_to_proj_data(*tmp_projdata_2d_sptr, add_scalar);

            norm_coeff_2d_sptr.reset(new BinNormalisationFromProjData(tmp_projdata_2d_sptr));
            norm_coeff_2d_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
            norm_coeff_2d_sptr->undo(*norm_projdata_2d_sptr, start_time, end_time);

            norm_coeff_2d_sptr.reset(new BinNormalisationFromProjData(norm_projdata_2d_sptr));
            norm_coeff_2d_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
        }
        else
        {
            norm_coeff_2d_sptr.reset(new BinNormalisationFromProjData(tmp_norm_projdata_sptr));
            norm_coeff_2d_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());
        }

    }

    this->multiplicative_binnorm_2d_sptr.reset(
                new ChainedBinNormalisation(norm_coeff_2d_sptr, atten_coeff_2d_sptr));
    this->multiplicative_binnorm_2d_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone());

    if (run_in_2d_projdata)
        iterative_object->get_objective_function_sptr()->set_normalisation_sptr(multiplicative_binnorm_2d_sptr);
    else
        iterative_object->get_objective_function_sptr()->set_normalisation_sptr(multiplicative_binnorm_sptr);

    info("ScatterEstimation: Done on normalisation coefficients.");

    //
    // Set additive (background) projdata
    //

    if (!is_null_ptr(this->add_projdata_sptr))
    {

        if( add_projdata_sptr->get_num_segments() > 1)
        {

            add_projdata_3d_sptr = add_projdata_sptr;
            info("ScatterEstimation: Running SSRB on the background data ...");

            FilePath tmp(this->back_projdata_filename);
            std::string out_filename = extras_path.get_path() + tmp.get_filename_no_extension() + "_2d.hs";

            this->add_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                   this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                   out_filename,
                                                                   std::ios::in | std::ios::out | std::ios::trunc));
            SSRB(*this->add_projdata_2d_sptr,
                 *this->add_projdata_3d_sptr, false);
            {
                std::string out_filename = extras_path.get_path() + "tmp_background_data.hs";

                this->back_projdata_3d_sptr.reset(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                        this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                        out_filename,
                                                                        std::ios::in | std::ios::out | std::ios::trunc));

                this->back_projdata_3d_sptr->fill(*this->add_projdata_3d_sptr);
            }
            {
                std::string out_filename = extras_path.get_path() + "tmp_background_data" + "_2d.hs";

                this->back_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                        this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                        out_filename,
                                                                        std::ios::in | std::ios::out | std::ios::trunc));
                this->back_projdata_2d_sptr->fill(*this->add_projdata_2d_sptr);
                this->multiplicative_binnorm_2d_sptr->apply(*this->back_projdata_2d_sptr, start_time, end_time);
            }

        }
        else
        {
            this->add_projdata_2d_sptr = add_projdata_sptr;
        }

        // Add the additive component to the output sinogram
        //                iterative_object->get_objective_function_sptr()->set_additive_proj_data_sptr(this->back_projdata_2d_sptr);
        //this->back_projdata_2d_sptr->fill(*this->add_projdata_2d_sptr);
//        back_projdata_2d_sptr->fill(*add_projdata_2d_sptr);
        //this->multiplicative_binnorm_2d_sptr->apply(*this->back_projdata_2d_sptr, start_time, end_time);

    }
    else
    {
        if (run_in_2d_projdata)
        {
            std::string out_filename = extras_path.get_path() + "tmp_background_data" + "_2d.hs";

            this->back_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                    this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                    out_filename,
                                                                    std::ios::in | std::ios::out | std::ios::trunc));
            this->back_projdata_2d_sptr->fill(0.0f);
        }

        std::string out_filename = extras_path.get_path() + "tmp_background_data.hs";

        this->back_projdata_3d_sptr.reset(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                out_filename,
                                                                std::ios::in | std::ios::out | std::ios::trunc));
        this->back_projdata_3d_sptr->fill(0.0f);
    }

    if (run_in_2d_projdata)
        back_projdata_sptr = this->back_projdata_2d_sptr;
    else
        back_projdata_sptr = this->back_projdata_3d_sptr;

    iterative_object->get_objective_function_sptr()->set_additive_proj_data_sptr(this->back_projdata_sptr);

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

    if (this->set_up() == Succeeded::no)
    {
        info("ScatterEstimation: Unsuccessful set up!");
       return Succeeded::no;
    }

    const double start_time = this->input_projdata_sptr->get_exam_info_sptr()->get_time_frame_definitions().get_start_time();
    const double end_time =this->input_projdata_sptr->get_exam_info_sptr()->get_time_frame_definitions().get_end_time();

    float local_min_scale_value = 0.5f;
    float local_max_scale_value = 0.5f;

    stir::BSpline::BSplineType  spline_type = stir::BSpline::quadratic;

    shared_ptr <ProjData> unscaled_est_projdata_sptr(new ProjDataInMemory(this->scatter_simulation_sptr->get_ExamInfo_sptr(),
                                                                          this->scatter_simulation_sptr->get_template_proj_data_info_sptr()->create_shared_clone()));
    scatter_simulation_sptr->set_output_proj_data_sptr(unscaled_est_projdata_sptr);

    shared_ptr<ProjData> scaled_est_projdata_sptr;
    shared_ptr<ProjData> data_to_fit_projdata_sptr;

    if(run_in_2d_projdata)
    {
        scaled_est_projdata_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                            this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone()));
        scaled_est_projdata_sptr->fill(0.F);

        std::string out_filename = extras_path.get_path() + "tmp_fit_data" + "_2d.hs";

        data_to_fit_projdata_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                              this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                              out_filename,
                                                              std::ios::in | std::ios::out | std::ios::trunc));
        data_to_fit_projdata_sptr->fill(*input_projdata_2d_sptr);
        //Data to fit = Input_2d - background
        //Here should be the total_background, not just the randoms.
        subtract_proj_data(*data_to_fit_projdata_sptr, *this->back_projdata_2d_sptr);
    }
    else
    {
        scaled_est_projdata_sptr.reset(new ProjDataInMemory(this->input_projdata_sptr->get_exam_info_sptr(),
                                                            this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone()));
        scaled_est_projdata_sptr->fill(0.F);

        std::string out_filename = extras_path.get_path() + "tmp_fit_data.hs";
        data_to_fit_projdata_sptr.reset(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                              this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                              out_filename,
                                                              std::ios::in | std::ios::out | std::ios::trunc));
        data_to_fit_projdata_sptr->fill(*input_projdata_sptr);

        subtract_proj_data(*data_to_fit_projdata_sptr, *this->back_projdata_3d_sptr);
    }

    info("ScatterEstimation: Start processing...");
    shared_ptr<DiscretisedDensity <3,float> > act_image_for_averaging;

    //Recompute the initial activity image if the max is equal to the min.
#if SPEED_UP_FOR_DEBUG == 0
    if( this->current_activity_image_sptr->find_max() == this->current_activity_image_sptr->find_min() )
    {
        info("ScatterEstimation: The max and the min values of the current activity image are equal."
             "We deduct that it has been initialised to some value, therefore we will run the intial "
             "recontruction ...");

        if (iterative_method)
            reconstruct_iterative(0, this->current_activity_image_sptr);
        else
            reconstruct_analytic(0, this->current_activity_image_sptr);

        if ( run_debug_mode )
        {
            std::string out_filename = extras_path.get_path() + "inital_activity_image";
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
        if (i_scat_iter > 0)
        {
            local_max_scale_value = this->max_scale_value;
            local_min_scale_value = this->min_scale_value;
        }

        scaled_est_projdata_sptr->fill(0.F);

        upsample_and_fit_scatter_estimate(*scaled_est_projdata_sptr, *data_to_fit_projdata_sptr,
                                          *unscaled_est_projdata_sptr,
                                          *this->multiplicative_binnorm_sptr->get_first_norm(),
                                          *this->mask_projdata_sptr, local_min_scale_value,
                                          local_max_scale_value, this->half_filter_width,
                                          spline_type, true);

        if(this->run_debug_mode)
        {
            std::stringstream convert;   // stream used for the conversion
            convert << "scaled_" << i_scat_iter;
            FilePath tmp(convert.str(),false);
            tmp.prepend_directory_name(extras_path.get_path());
            dynamic_cast<ProjDataInMemory *> (scaled_est_projdata_sptr.get())->write_to_file(tmp.get_string());
        }

        if (this->export_scatter_estimates_of_each_iteration ||
                i_scat_iter == this->num_scatter_iterations -1 )
        {

            //this is complicated as the 2d scatter estimate was
            //divided by norm2d, so we need to undo this
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

            // ok, we can multiply with the norm
            this->multiplicative_binnorm_sptr->apply_only_first(*temp_projdata, start_time, end_time);

            std::stringstream convert;   // stream used for the conversion
            convert << this->output_scatter_estimate_prefix << "_" <<
                       i_scat_iter;
            std::string output_scatter_filename = convert.str();

            // To save the 3d scatter estimate
            shared_ptr <ProjData> temp_scatter_projdata_3d (
                        new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                              this->input_projdata_sptr->get_proj_data_info_sptr() ,
                                              output_scatter_filename,
                                              std::ios::in | std::ios::out | std::ios::trunc));
            shared_ptr<BinNormalisation> dummy_normalisation_coeffs_3d_sptr(new TrivialBinNormalisation());
            // Upsample to 3D
            //we're currently not doing the tail fitting in this step, but keeping the same scale as determined in 2D

            upsample_and_fit_scatter_estimate(*temp_scatter_projdata_3d,
                                              *this->input_projdata_sptr,
                                              *temp_projdata,
                                              *dummy_normalisation_coeffs_3d_sptr,
                                              *this->input_projdata_sptr,
                                              1.0f, 1.0f, 1, spline_type,
                                              false);

            // Now save the full background term.
            convert.clear();
            convert << this->output_background_estimate_prefix << "_" <<
                       i_scat_iter;
            std::string output_background_filename = convert.str();

            shared_ptr <ProjData> temp_background_projdata_3d (
                        new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                              this->input_projdata_sptr->get_proj_data_info_sptr() ,
                                              output_background_filename,
                                              std::ios::in | std::ios::out | std::ios::trunc));
            temp_background_projdata_3d->fill(*temp_scatter_projdata_3d);

            if (!is_null_ptr(add_projdata_3d_sptr))
                add_proj_data(*temp_background_projdata_3d, *this->add_projdata_3d_sptr);
            this->multiplicative_binnorm_3d_sptr->apply(*temp_background_projdata_3d, start_time, end_time);
        }

        this->back_projdata_sptr->fill(*scaled_est_projdata_sptr);
        if (!is_null_ptr(add_projdata_sptr))
            add_proj_data(*back_projdata_sptr, *this->add_projdata_2d_sptr);

        this->multiplicative_binnorm_sptr->apply(*back_projdata_sptr, start_time, end_time);

        if(run_in_2d_projdata)
        {
            data_to_fit_projdata_sptr->fill(*input_projdata_2d_sptr);
            subtract_proj_data(*data_to_fit_projdata_sptr, *this->back_projdata_2d_sptr);
        }
        else {
            data_to_fit_projdata_sptr->fill(*input_projdata_sptr);
            subtract_proj_data(*data_to_fit_projdata_sptr, *this->back_projdata_sptr);
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

    shared_ptr<IterativeReconstruction<DiscretisedDensity<3, float> >> tmp_iterative =
            dynamic_pointer_cast<IterativeReconstruction<DiscretisedDensity<3, float> > >(reconstruction_template_sptr);

    //
    // Now, we can call Reconstruction::set_up().
    if (tmp_iterative->set_up(this->current_activity_image_sptr) == Succeeded::no)
    {
        error("ScatterEstimation: Failure at set_up() of the reconstruction method. Aborting.");
    }

    tmp_iterative->set_start_subset_num(0);
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
ScatterEstimation::write_log() const
{
    //    std::string log_filename =
    //            this->output_proj_data_filename + ".log";
    //    std::ofstream mystream(log_filename.c_str());

    //    if (!mystream)
    //    {
    //        warning("Cannot open log file '%s'", log_filename.c_str()) ;
    //        return;
    //    }

    //    int axial_bins = 0 ;

    //    for (int segment_num = this->output_proj_data_sptr->get_min_segment_num();
    //         segment_num <= this->output_proj_data_sptr->get_max_segment_num();
    //         ++segment_num)
    //        axial_bins += this->output_proj_data_sptr->get_num_axial_poss(segment_num);

    //    const int total_bins =
    //            this->output_proj_data_sptr->get_num_views() * axial_bins *
    //            this->output_proj_data_sptr->get_num_tangential_poss();
    //    mystream << this->parameter_info()
    //             << "\nTotal simulation time elapsed: "
    //             <<   simulation_time / 60 << "min"
    //               << "\nTotal Scatter Points : " << scatt_points_vector.size()
    //               << "\nTotal Scatter Counts : " << total_scatter
    //               << "\nActivity image SIZE: "
    //               << (*this->activity_image_sptr).size() << " * "
    //               << (*this->activity_image_sptr)[0].size() << " * "  // TODO relies on 0 index
    //               << (*this->activity_image_sptr)[0][0].size()
    //            << "\nAttenuation image SIZE: "
    //            << (*this->atten_image_sptr).size() << " * "
    //            << (*this->atten_image_sptr)[0].size() << " * "
    //            << (*this->atten_image_sptr)[0][0].size()
    //            << "\nTotal bins : " << total_bins << " = "
    //            << this->output_proj_data_sptr->get_num_views()
    //            << " view_bins * "
    //            << axial_bins << " axial_bins * "
    //            << this->output_proj_data_sptr->get_num_tangential_poss()
    //            << " tangential_bins\n";
}

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
ScatterEstimation::ffw_project_mask_image()
{
    if (is_null_ptr(this->mask_image_sptr))
    {
        warning("You cannot forward project if you have not set the mask image. Aborting.");
        return Succeeded::no;
    }

    if (is_null_ptr(this->input_projdata_2d_sptr))
    {
        warning("No 2D proj_data have been initialised. Aborting.");
        return Succeeded::no;
    }

    shared_ptr<ForwardProjectorByBin> forw_projector_sptr;
    shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
    forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    info(boost::format("ScatterEstimation: Forward projector used for the calculation of "
                       "the tail mask: %1%") % forw_projector_sptr->parameter_info());

    forw_projector_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_ptr()->create_shared_clone(),
                                this->mask_image_sptr );

    shared_ptr<ProjData> mask_projdata(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                            this->input_projdata_2d_sptr->get_proj_data_info_ptr()->create_shared_clone()));

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
                                                             mask_projdata->get_proj_data_info_ptr()->create_shared_clone(),
                                                             this->mask_projdata_filename,
                                                             std::ios::in | std::ios::out | std::ios::trunc));
    else
        this->mask_projdata_sptr.reset(new ProjDataInMemory(mask_projdata->get_exam_info_sptr(),
                                                            mask_projdata->get_proj_data_info_ptr()->create_shared_clone()));

    CreateTailMaskFromACFs create_tail_mask_from_acfs;

    if(!create_tail_mask_from_acfs.parse(this->tail_mask_par_filename.c_str()))
    {
        warning(boost::format("Error parsing parameters file %1%, for creating mask tails from ACFs. Setting up to default.")
                %this->tail_mask_par_filename);
        //return Succeeded::no;
        create_tail_mask_from_acfs.ACF_threshold = 1.1;
        create_tail_mask_from_acfs.safety_margin = 4;
    }

    create_tail_mask_from_acfs.set_input_projdata_sptr(mask_projdata);
    create_tail_mask_from_acfs.set_output_projdata_sptr(this->mask_projdata_sptr);
    return create_tail_mask_from_acfs.process_data();
}

//! If the filters are not applied in this specific order the
//! results are not the desirable every time.
bool
ScatterEstimation::
apply_mask_in_place(DiscretisedDensity<3, float>& arg,
                    const mask_parameters& _this_mask)
{
    // Re-reading every time should not be a problem, as it is
    // just a small txt file.
    PostFiltering<DiscretisedDensity<3, float> > filter;

    if(filter.parse(this->mask_postfilter_filename.c_str()) == false)
    {
        warning(boost::format("Error parsing postfilter parameters file %1%. Aborting.")
                %this->mask_postfilter_filename);
        return false;
    }

    //1. add_scalar//2. mult_scalar//3. power//4. min_threshold//5. max_threshold

    pow_times_add pow_times_thres_max(0.0f, 1.0f, 1.0f, NumericInfo<float>().min_value(),
                                      0.001f);
    pow_times_add pow_times_add_scalar( -0.00099f, 1.0f, 1.0f, NumericInfo<float>().min_value(),
                                        NumericInfo<float>().max_value());

    pow_times_add pow_times_thres_min(0.0, 1.0f, 1.0f, 0.0f,
                                      NumericInfo<float>().max_value());

    pow_times_add pow_times_times( 0.0f, 100002.0f, 1.0f, NumericInfo<float>().min_value(),
                                   NumericInfo<float>().max_value());

    pow_times_add pow_times_add_object(_this_mask.add_scalar,
                                       _this_mask.times_scalar,
                                       1.0f,
                                       _this_mask.min_threshold,
                                       _this_mask.max_threshold);
    // 1. filter the image
    filter.process_data(arg);

    // 2. max threshold
    in_place_apply_function(arg,
                            pow_times_thres_max);
    // 3. add scalar
    in_place_apply_function(arg,
                            pow_times_add_scalar);
    // 4. min threshold
    in_place_apply_function(arg,
                            pow_times_thres_min);
    // 5. times scalar
    in_place_apply_function(arg,
                            pow_times_times);
    // 6. Add 1.
    in_place_apply_function(arg,
                            pow_times_add_object);

    return true;
}

int ScatterEstimation::get_iterations_num() const
{
    return num_scatter_iterations;
}

Succeeded
ScatterEstimation::prepare_projdata(const shared_ptr<ProjData> input,
                                    shared_ptr<ProjData> output)
{

}

END_NAMESPACE_STIR
