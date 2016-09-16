/*
  Copyright (C) 2004 -  2009 Hammersmith Imanet Ltd
  Copyright (C) 2013 - 2016 University College London
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
  \brief Implementation of most functions in stir::ScatterEstimationByBin

  \author Nikos Efthimiou
  \author Kris Thielemans
*/
#include "stir/scatter/ScatterEstimationByBin.h"
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
#include "stir/stir_math.h"
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

void
ScatterEstimationByBin::
set_defaults()
{
    // All recomputes default true
    this->recompute_initial_activity_image = true;
    this->recompute_atten_projdata = true;
    this->recompute_mask_atten_image = true;
    this->recompute_mask_projdata = true;
    this->initial_activity_image_filename = "";
    this->atten_image_filename = "";
    this->o_scatter_estimate_prefix = "";
    this->num_scatter_iterations = 5;
    this->min_scale_value = 0.4f;
    this->max_scale_value = 100.f;
    this->half_filter_width = 3;
    this->iterative_method = true;
    this->do_average_at_2 = true;
    this->export_scatter_estimates_of_each_iteration = false;
    this->run_debug_mode = false;
    this->override_initial_activity_image = false;
    this->override_density_image = false;
    this->override_density_image_for_scatter_points = false;
}

void
ScatterEstimationByBin::
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
                         &this->mask_atten_image_filename);
    this->parser.add_key("mask postfilter filename",
                         &this->mask_postfilter_filename);
    //-> for attenuation
    this->parser.add_key("recompute mask attenuation image",
                         &this->recompute_mask_atten_image);
    this->parser.add_key("mask attenuation max threshold ",
                         &this->mask_attenuation_image.max_threshold);
    this->parser.add_key("mask attenuation add scalar",
                         &this->mask_attenuation_image.add_scalar);
    this->parser.add_key("mask attenuation min threshold",
                         &this->mask_attenuation_image.min_threshold);
    this->parser.add_key("mask attenuation times scalar",
                         &this->mask_attenuation_image.times_scalar);
    // MASK PROJDATA
    this->parser.add_key("recompute mask projdata",
                         &this->recompute_mask_projdata);
    this->parser.add_key("mask projdata filename",
                         &this->mask_projdata_filename);
    this->parser.add_key("tail fitting par filename",
                         &this->tail_mask_par_filename);
    // END MASK PROJDATA

    this->parser.add_key("attenuation projdata filename",
                         &this->atten_coeff_filename);
    this->parser.add_key("recompute attenuation projdata",
                         &this->recompute_atten_projdata);

    this->parser.add_key("background projdata filename",
                         &this->back_projdata_filename);
    this->parser.add_key("normalisation projdata filename",
                         &this->norm_projdata_filename);

    this->parser.add_key("recompute initial activity image",
                         &this->recompute_initial_activity_image);
    this->parser.add_key("initial activity image filename",
                         &this->initial_activity_image_filename);

    // RECONSTRUCTION RELATED
    this->parser.add_key("reconstruction template file",
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

    this->parser.add_key("over-ride initial activity image",
                         &this->override_initial_activity_image);
    this->parser.add_key("over-ride density image",
                         &this->override_density_image);
    this->parser.add_key("over-ride density image for scatter points",
                         &this->override_density_image_for_scatter_points);
    // END Scatter simulation

    this->parser.add_key("export scatter estimates of each iteration",
                         &this->export_scatter_estimates_of_each_iteration);
    this->parser.add_key("output scatter estimate name prefix",
                         &this->o_scatter_estimate_prefix);
    this->parser.add_key("do average at 2",
                         &this->do_average_at_2);
    this->parser.add_key("max scale value",
                         &this->max_scale_value);
    this->parser.add_key("min scale value",
                         &this->min_scale_value);
    this->parser.add_key("half filter width",
                         &this->half_filter_width);
    this->parser.add_key("remove interleaving",
                         &this->remove_interleaving);
    this->parser.add_key("export 2s projdata",
                         &this->export_2d_projdata);
}

ScatterEstimationByBin::
ScatterEstimationByBin()
{
    this->set_defaults();
}

bool
ScatterEstimationByBin::
post_processing()
{
    // Check that the crusial parts have been set.
    info("Loading input projection data");
    if (this->input_projdata_filename.size() == 0)
    {
        warning("No input projdata filename is given. Abort ");
        return true;
    }

    this->input_projdata_sptr =
            ProjData::read_from_file(this->input_projdata_filename);

    // If the reconstruction_template_sptr is null then, we need to parse it from another
    // file. I prefer this implementation since makes smaller modular files.
    if (this->recon_template_par_filename.size() == 0)
    {
        warning("Please define a reconstruction method. Abord.");
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
            warning(boost::format("Error parsing reconstruction parameters file %1%. Abord.")
                    %this->recon_template_par_filename);
            return true;
        }
    }

    info("Loading attenuation image...");
    if (this->atten_image_filename.size() == 0)
    {
        warning("Please define an attenuation image. Abord.");
        return true;
    }
    else
        this->atten_image_sptr =
            read_from_file<DiscretisedDensity<3,float> >(this->atten_image_filename);

    if(this->atten_coeff_filename.size() > 0)
    {
        if (!this->recompute_atten_projdata)
        {
            info("Loading attenuation correction coefficients...");
            this->atten_projdata_sptr =
                    ProjData::read_from_file(this->atten_coeff_filename);
        }
    }

    if (this->norm_projdata_filename.size() > 0 )
    {
        info("Loading normalisation coefficients...");
        this->norm_projdata_sptr =
                ProjData::read_from_file(this->norm_projdata_filename);
    }

    if (this->back_projdata_filename.size() > 0)
    {
        info("Loading background projdata...");
        this->back_projdata_sptr =
                ProjData::read_from_file(this->back_projdata_filename);
    }

    if(!this->recompute_initial_activity_image ) // This image can be used as a template
    {
        info("Loading initial activity image ...");
        if(this->initial_activity_image_filename.size() > 0 )
            this->activity_image_lowres_sptr =
                read_from_file<DiscretisedDensity<3,float> >(this->initial_activity_image_filename);
        else
        {
            warning("Recompute initial activity image was set to false but"
                    "no file name was set. Abord.");
            return true;
        }
    }

    info ("Initialising mask image ... ");
    if(this->mask_postfilter_filename.size() == 0)
        return true;
    else
    {
        this->filter_sptr.reset(new PostFiltering <DiscretisedDensity<3,float> >);

        if(!filter_sptr->parse(this->mask_postfilter_filename.c_str()))
        {
            warning(boost::format("Error parsing post filter parameters file %1%. Abord.")
                    %this->mask_postfilter_filename);
            return true;
        }
    }

    info ("Initialising Scatter Simulation ... ");
    if (this->scatter_sim_par_filename.size() == 0)
    {
        warning("Please define a scatter simulation method. Abord.");
        return true;
    }
    else
    {
        KeyParser local_parser;
        local_parser.add_start_key("Scatter Simulation");
        local_parser.add_stop_key("End Scatter Simulation");
        local_parser.add_parsing_key("Simulation method", &this->scatter_simulation_sptr);
        if (!local_parser.parse(this->scatter_sim_par_filename.c_str()))
        {
            warning(boost::format("Error parsing scatter simulation parameters file %1%. Abord.")
                    %this->recon_template_par_filename);
            return true;
        }
    }

    if (this->o_scatter_estimate_prefix.size() == 0)
        return true;

    if(!this->recompute_mask_projdata)
    {
        if (this->mask_projdata_filename.size() == 0)
        {
            warning("Please define a filename for mask proj_data. Abord.");
            return true;
        }
        this->mask_projdata_sptr =
                ProjData::read_from_file(this->mask_projdata_filename);
    }
    else
    {
        if (!this->recompute_mask_atten_image)
        {
            if (this->mask_atten_image_filename.size() == 0 )
            {
                warning("Please define a filename for mask image. Abord.");
                return true;
            }

            this->mask_atten_image_sptr =
                    read_from_file<DiscretisedDensity<3, float> >(this->mask_atten_image_filename);
        }

        if (this->tail_mask_par_filename.size() == 0)
        {
            warning("Please define a filename for tails mask. Abord.");
            return true;
        }
    }

    return false;
}

Succeeded
ScatterEstimationByBin::
set_up()
{
    if (this->run_debug_mode)
    {
        info("Debugging mode is activated.");
        this->export_2d_projdata = true;
        this->export_scatter_estimates_of_each_iteration = true;
    }

    if (is_null_ptr(this->input_projdata_sptr))
    {
        warning(boost::format("Input projdata file %1% not found. Abord.") %this->input_projdata_filename );
        return Succeeded::no;
    }

    // Load InputProjData and calculate the SSRB
    if (this->input_projdata_sptr->get_max_segment_num() > 0 )
    {
        this->proj_data_info_2d_sptr.reset(
                    dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >
                    (SSRB(*this->input_projdata_sptr->get_proj_data_info_ptr(),
                          this->input_projdata_sptr->get_max_segment_num())));

        if (this->export_2d_projdata)
        {
            size_t lastindex = this->input_projdata_filename.find_last_of(".");
            std::string rawname = this->input_projdata_filename.substr(0, lastindex);
            std::string out_filename = rawname + "_2d.hs";

            this->input_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                     this->proj_data_info_2d_sptr,
                                                                     out_filename,
                                                                     std::ios::in | std::ios::out | std::ios::trunc));
        }
        else
            this->input_projdata_2d_sptr.reset(new ProjDataInMemory(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                    this->proj_data_info_2d_sptr));

        SSRB(*this->input_projdata_2d_sptr,
             *input_projdata_sptr,false);
    }
    else
    {
        warning(boost::format("The input data %1% are not 3D. Abord.") %this->input_projdata_filename );
        return Succeeded::no;
    }

    //
    // Load the initial activity image if !recomput_initial_activity_image, else
    // Create and initialise the activity image (and run reconstruction::set_up())
    //
    if (!this->recompute_initial_activity_image && this->initial_activity_image_filename.size() > 0 )
    {
        info("Loading initial activity image ...");
        this->activity_image_sptr =
                read_from_file<DiscretisedDensity<3,float> >(this->initial_activity_image_filename);
    }
    else
    {
        info("Initialising empty activity image ... ");
        this->activity_image_sptr.reset(new VoxelsOnCartesianGrid<float> ( *this->input_projdata_2d_sptr->get_proj_data_info_sptr() ) );
        this->activity_image_sptr->fill(1.F);
    }

    //
    // Beside the attenuation correction we need it for the mask and the
    // ScatterSimulation
    //
    {
        if (is_null_ptr(this->atten_image_sptr))
        {
            warning("Attenuation image has not been loaded properly. Abord.");
            return Succeeded::no;
        }

        info(boost::format("Attenuation image data are supposed to be in units cm^-1\n"
                           "\tReference: water has mu .096 cm^-1\n"
                           "\tMax in attenuation image: %g\n") %
             this->atten_image_sptr->find_max());

        int min_z = this->atten_image_sptr->get_min_index();
        int min_y = this->atten_image_sptr.get()[0][min_z].get_min_index();
        int len_y = this->atten_image_sptr.get()[0][min_z].get_length();
        int len_x = this->atten_image_sptr.get()[0][min_z][min_y].get_length();

        if (len_y != len_x)
            error(boost::format("The voxels in the x (%1%) and y (%2%)"
                                " are different. Cannot zoom...  ") %len_x % len_y );
    }

    //
    // Zoom the activity image
    // TODO: add option to opt-out
    //
    {
        int size_xy = 0;
        int size_z = 0;
        // zoom image
        float zoom_z = 1.0f;
        float zoom_xy = 0.3f;

        if(!is_null_ptr(this->activity_image_lowres_sptr))
        {
            info("Zooming attenuation image to initial activity image...");

            VoxelsOnCartesianGrid<float>* tmp_image_ptr =
                    dynamic_cast<VoxelsOnCartesianGrid<float>* >(this->activity_image_lowres_sptr.get());

            size_xy = tmp_image_ptr->get_x_size();
            size_z = tmp_image_ptr->get_z_size();

        }
        else
        {
            VoxelsOnCartesianGrid<float>* tmp_image_ptr =
                    dynamic_cast<VoxelsOnCartesianGrid<float>* >(this->activity_image_sptr.get());

            size_xy = static_cast<int>(tmp_image_ptr->get_x_size() * zoom_xy );
            size_z = static_cast<int>(tmp_image_ptr->get_z_size() * zoom_z) ;

            VoxelsOnCartesianGrid<float> activity_lowres =
                    zoom_image(*tmp_image_ptr,
                               CartesianCoordinate3D<float>(zoom_z, zoom_xy, zoom_xy),
                               CartesianCoordinate3D<float>(0.0f, 0.0f, 0.0f),
                               CartesianCoordinate3D<int>(size_z, size_xy, size_xy));

            this->activity_image_lowres_sptr.reset(activity_lowres.clone());
            this->activity_image_lowres_sptr->fill(1.f);
        }

        info("Zooming attenuation image to match the activity image");

        VoxelsOnCartesianGrid<float>* attenuation_image_ptr =
                dynamic_cast<VoxelsOnCartesianGrid<float>* >(this->atten_image_sptr.get());
        shared_ptr< DiscretisedDensity<3, float> > attenuation_lowres(this->activity_image_lowres_sptr->get_empty_copy());
        VoxelsOnCartesianGrid<float>* attenuation_lowres_ptr =
                dynamic_cast<VoxelsOnCartesianGrid<float>* >(attenuation_lowres.get());

        zoom_image(*attenuation_lowres_ptr, *attenuation_image_ptr);

        if(this->filter_sptr->is_filter_null())
        {
            warning(boost::format("Error creating a filter from %1%. Abord.") %this->mask_postfilter_filename );
            return Succeeded::no;
        }

        if (this->filter_sptr->process_data(*attenuation_lowres_ptr) == Succeeded::no)
            return Succeeded::no;

        float scale_att = zoom_xy * zoom_xy * zoom_z;

        // Just multiply with the scale factor.
        pow_times_add scale_factor(0.0f, scale_att, 1.0f,
                                   NumericInfo<float>().min_value(),
                                   NumericInfo<float>().max_value());

        in_place_apply_function(*attenuation_lowres_ptr, scale_factor);

        this->atten_image_lowres_sptr.reset(attenuation_lowres->clone());

        if(is_null_ptr(this->reconstruction_template_sptr))
        {
            warning("Reconstruction method has not been initialised. Abord.");
            return Succeeded::no;
        }
    }

    // Allocate the output projdata.
    this->scaled_est_projdata_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                              this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone()));

    this->scaled_est_projdata_sptr->fill(0.F);

    info("Setting up reconstruction method ...");
    {
        AnalyticReconstruction* tmp_analytic =
                dynamic_cast<AnalyticReconstruction * >(this->reconstruction_template_sptr.get());

        if (!is_null_ptr(tmp_analytic))
            if(set_up_analytic() == Succeeded::no)
                error("Initialisation failed!");

        this->iterative_method = false;
    }

    {
        IterativeReconstruction<DiscretisedDensity<3, float> >* tmp_iterative =
                dynamic_cast<IterativeReconstruction<DiscretisedDensity<3, float> > * >(this->reconstruction_template_sptr.get());
        if (!is_null_ptr(tmp_iterative))
            if(set_up_iterative(tmp_iterative) == Succeeded::no)
                error("Initialisation failed!");

        this->iterative_method = true;
    }

    //
    // Now, we can call set up reconstuction.
    if (this->reconstruction_template_sptr->set_up(this->activity_image_lowres_sptr) == Succeeded::no)
    {
        warning ("Failure is set_up() of the reconstruction method. Abord.");
        return Succeeded::no;
    }

    //
    // ScatterSimulation
    //

    info("Setting up Scatter Simulation method ...");
    if(is_null_ptr(this->scatter_simulation_sptr))
    {
        warning("Scatter simulation method has not been initialised. Abord.");
        return Succeeded::no;
    }

    // The images are passed to the simulation.
    // and it will override anything that the ScatterSimulation.par file has done.
    if(this->override_density_image)
        this->scatter_simulation_sptr->set_density_image_sptr(this->atten_image_lowres_sptr);
    if(this->override_density_image_for_scatter_points)
        this->scatter_simulation_sptr->set_density_image_for_scatter_points_sptr(this->atten_image_lowres_sptr);
    if(this->override_initial_activity_image)
        this->scatter_simulation_sptr->set_activity_image_sptr(this->activity_image_lowres_sptr);

    // Check if Load a mask proj_data

    if(is_null_ptr(this->mask_projdata_sptr))
    {
        if(is_null_ptr(this->mask_atten_image_sptr))
        {
            // Applying mask
            // 1. Clone from the original image.
            // 2. Apply to the new clone.
            this->mask_atten_image_sptr.reset(this->atten_image_sptr->clone());
            if(this->apply_mask_in_place(this->mask_atten_image_sptr,
                                         this->mask_attenuation_image) == false)
            {
                warning("Error in masking. Abord.");
                return Succeeded::no;
            }

            if (this->mask_atten_image_filename.size() > 0 )
                OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                        write_to_file(this->mask_atten_image_filename, *this->mask_atten_image_sptr.get());
        }

        if(ffw_project_mask_image() == Succeeded::no)
        {
            warning("Unsuccessfull to fwd project the mask image. Abord.");
            return Succeeded::no;
        }
    }

    info(">>>>Set up finished successfully!!<<<<");
    return Succeeded::yes;
}

Succeeded
ScatterEstimationByBin::
set_up_iterative(IterativeReconstruction<DiscretisedDensity<3, float> > * iterative_object)
{
    info("Setting up iterative reconstruction ...");
    iterative_object->set_input_data(this->input_projdata_2d_sptr);

    //
    // Multiplicative projdata
    //

    shared_ptr<BinNormalisation> attenuation_correction(new TrivialBinNormalisation());
    shared_ptr<BinNormalisation> normalisation_coeffs(new TrivialBinNormalisation());

    if (is_null_ptr(this->atten_projdata_sptr))
    {
        warning("No attenuation projdata have been initialised. "
                "Forward Projecting the attenuation image... ");

        //Project the attenuation imge.
        shared_ptr<ForwardProjectorByBin> forw_projector_sptr;
        shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
        forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
        info(boost::format("\n\nForward projector used for the calculation of\n"
                           "attenuation coefficients: %1%\n") % forw_projector_sptr->parameter_info());
        forw_projector_sptr->set_up(this->input_projdata_sptr->get_proj_data_info_ptr()->create_shared_clone(),
                                    this->atten_image_sptr );

        // I think I could project directly to 2D ...
        // - No, I would still need the 3D version at the very end.
        if (this->atten_coeff_filename.size() > 0)
            this->atten_projdata_sptr.reset(new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                  this->input_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone(),
                                                                  this->atten_coeff_filename,
                                                                  std::ios::in | std::ios::out | std::ios::trunc));
        else
            this->atten_projdata_sptr.reset(new ProjDataInMemory(this->input_projdata_sptr->get_exam_info_sptr(),
                                                                 this->input_projdata_sptr->get_proj_data_info_ptr()->create_shared_clone()));
        forw_projector_sptr->forward_project(* this->atten_projdata_sptr, *this->atten_image_sptr);
    }

    if( this->atten_projdata_sptr->get_max_segment_num() > 0)
    {
        info("Running SSRB on attenuation correction coefficients ...");
        if(this->export_2d_projdata)
        {
            size_t lastindex = this->atten_coeff_filename.find_last_of(".");
            std::string rawname = this->atten_coeff_filename.substr(0, lastindex);
            std::string out_filename = rawname + "_2d.hs";

            this->atten_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                     this->proj_data_info_2d_sptr,
                                                                     out_filename,
                                                                     std::ios::in | std::ios::out | std::ios::trunc));
        }
        else
            this->atten_projdata_2d_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                    this->proj_data_info_2d_sptr));

        SSRB(*this->atten_projdata_2d_sptr,
             *this->atten_projdata_sptr, true);
    }
    else
    {
        error(boost::format("The attenuation projdata %1% are not 3D") % this->atten_coeff_filename);
    }

    attenuation_correction.reset(new BinNormalisationFromProjData(this->atten_projdata_2d_sptr));

    //<- End of Attenuation projdata

    // Normalisation ProjData
    if (!is_null_ptr(this->norm_projdata_sptr))
    {
        if( this->norm_projdata_sptr->get_max_segment_num() > 0) //If the sinogram is 3D then SSRB it.
        {
            info("Performing SSRB on normalisation coefficients ... ");
            shared_ptr < ProjData> inv_norm_projdata_sptr(new ProjDataInMemory(this->norm_projdata_sptr->get_exam_info_sptr(),
                                                                               this->norm_projdata_sptr->get_proj_data_info_ptr()->create_shared_clone()));
            inv_norm_projdata_sptr->fill(0.f);
            // We need to get norm2d=1/SSRB(1/norm3d))
            pow_times_add inverse_object(0.0f, 1.0f, -1.0f, NumericInfo<float>().min_value(),
                                         NumericInfo<float>().max_value());

            for (int segment_num = inv_norm_projdata_sptr->get_min_segment_num();
                 segment_num <= inv_norm_projdata_sptr->get_max_segment_num();
                 ++segment_num)
            {
                SegmentByView<float> segment_by_view =
                        this->norm_projdata_sptr->get_segment_by_view(segment_num);

                in_place_apply_function(segment_by_view,
                                        inverse_object);

                if (!(inv_norm_projdata_sptr->set_segment(segment_by_view) == Succeeded::yes))
                {
                    warning("Error set_segment %d\n", segment_num);
                    return Succeeded::no;
                }
            }

            if (this->export_2d_projdata)
            {
                size_t lastindex = this->norm_projdata_filename.find_last_of(".");
                std::string rawname = this->norm_projdata_filename.substr(0, lastindex);
                std::string out_filename = rawname + "_2d.hs";

                this->norm_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                        this->proj_data_info_2d_sptr,
                                                                        out_filename,
                                                                        std::ios::in | std::ios::out | std::ios::trunc));
            }
            else
                this->norm_projdata_2d_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                       this->proj_data_info_2d_sptr));

            SSRB(*this->norm_projdata_2d_sptr,
                 *inv_norm_projdata_sptr,false);

            for (int segment_num = this->norm_projdata_2d_sptr->get_min_segment_num();
                 segment_num <= this->norm_projdata_2d_sptr->get_max_segment_num();
                 ++segment_num)
            {
                SegmentByView<float> segment_by_view =
                        this->norm_projdata_2d_sptr->get_segment_by_view(segment_num);

                in_place_apply_function(segment_by_view,
                                        inverse_object);

                if (!(this->norm_projdata_2d_sptr->set_segment(segment_by_view) == Succeeded::yes))
                {
                    warning("Error set_segment %d\n", segment_num);
                    return Succeeded::no;
                }
            }
        }
        else
        {
            error(boost::format("The normalisation projdata %1% are not 3D") % this->norm_projdata_filename);
        }

        normalisation_coeffs.reset(new BinNormalisationFromProjData(this->norm_projdata_2d_sptr));
    }
    //<- End Normalisation ProjData

    this->multiplicative_data_2d_sptr.reset(
                new ChainedBinNormalisation(attenuation_correction, normalisation_coeffs));

    iterative_object->get_objective_function_sptr()->set_normalisation_sptr(multiplicative_data_2d_sptr);

    //
    // Set additive (background) projdata
    //

    if (!is_null_ptr(this->back_projdata_sptr))
    {

        if( back_projdata_sptr->get_max_segment_num() > 0)
        {
            size_t lastindex = this->back_projdata_filename.find_last_of(".");
            std::string rawname = this->back_projdata_filename.substr(0, lastindex);
            std::string out_filename = rawname + "_2d.hs";

            this->back_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                                    this->proj_data_info_2d_sptr,
                                                                    out_filename,
                                                                    std::ios::in | std::ios::out | std::ios::trunc));

            SSRB(*this->back_projdata_2d_sptr,
                 *back_projdata_sptr, false);
        }
        else
        {
            this->back_projdata_2d_sptr = back_projdata_sptr;
        }

        // Add the additive component to the output sinogram
        //        iterative_object->set_additive_proj_data_sptr(this->back_projdata_2d_sptr);
        this->scaled_est_projdata_sptr->fill(*this->back_projdata_2d_sptr);

    }

    iterative_object->get_objective_function_sptr()->set_additive_proj_data_sptr(this->scaled_est_projdata_sptr);
    return Succeeded::yes;
}

Succeeded
ScatterEstimationByBin::
set_up_analytic()
{
    //TODO : I have most stuff in tmp.
    return Succeeded::yes;
}

Succeeded
ScatterEstimationByBin::
process_data()
{

    if (this->set_up() == Succeeded::no)
        return Succeeded::no;

    float local_min_scale_value = 0.5f;
    float local_max_scale_value = 0.5f;

    stir::BSpline::BSplineType  spline_type = stir::BSpline::linear;
    shared_ptr <ProjData> unscaled = this->scatter_simulation_sptr->get_output_proj_data_sptr();
    shared_ptr<BinNormalisation> empty(new TrivialBinNormalisation());
    shared_ptr<DiscretisedDensity <3,float> > act_image_for_averaging;

    if( this->recompute_initial_activity_image )
    {
        info("Computing initial activity image...");
        if (iterative_method)
            reconstruct_iterative( -1,
                                   this->activity_image_lowres_sptr);
        else
            reconstruct_analytic();

        if (this->initial_activity_image_filename.size() > 0 )
            OutputFileFormat<DiscretisedDensity < 3, float > >::default_sptr()->
                    write_to_file(this->initial_activity_image_filename, *this->activity_image_lowres_sptr);
    }

    if ( this->do_average_at_2)
        act_image_for_averaging.reset(this->activity_image_lowres_sptr->clone());

    //
    // Begin the estimation process...
    //
    for (int i_scat_iter = 0;
         i_scat_iter < this->num_scatter_iterations;
         i_scat_iter++)
    {

        if ( this->do_average_at_2)
        {
            if (i_scat_iter == 2) // do average 0 and 1
            {
                if (is_null_ptr(act_image_for_averaging))
                    error("Storing the first actibity estimate has failed at some point.");

                //                if (this->run_debug_mode)// Write the previous estimate
                //                {
                //                    std::string output1= "./extras/act_image_for_averaging";
                //                    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                //                            write_to_file(output1, *act_image_for_averaging);
                //                }
                std::transform(act_image_for_averaging->begin_all(), act_image_for_averaging->end_all(),
                               this->activity_image_lowres_sptr->begin_all(),
                               this->activity_image_lowres_sptr->begin_all(),
                               std::plus<float>());

                //                if(this->run_debug_mode)// write after sum
                //                {
                //                    std::string output1= "./extras/summed_for_averaging";
                //                    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                //                            write_to_file(output1, *this->activity_image_lowres_sptr);
                //                }

                pow_times_add divide_by_two (0.0f, 0.5f, 1.0f, NumericInfo<float>().min_value(),
                                             NumericInfo<float>().max_value());

                in_place_apply_function(*this->activity_image_lowres_sptr, divide_by_two);

                //                if(this->run_debug_mode)// write after division.
                //                {
                //                    std::string output1= "./extras/recon_0_1";
                //                    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                //                            write_to_file(output1, *this->activity_image_lowres_sptr);
                //                }

                // Have averaged ... turn false
                this->do_average_at_2 = false;
            }
        }

        //TODO: filter ... if FBP

        info("Scatter simulation in progress...");
        //        this->scatter_simulation_sptr->set_activity_image_sptr(this->activity_image_lowres_sptr);
        this->scatter_simulation_sptr->process_data();
        //        this->scatter_simulation_sptr->set_output_proj_data(unscaled);

        if(this->run_debug_mode) // Write unscaled scatter sinogram
        {
            std::stringstream convert;   // stream used for the conversion
            int lab = i_scat_iter+1;
            convert << "./extras/unscaled_" << lab;
            std::string output_filename =  convert.str();

            dynamic_cast<ProjDataInMemory *> (unscaled.get())->write_to_file(output_filename);
        }


        // Set the min and max scale factors
        if (i_scat_iter > 0)
        {
            local_max_scale_value = this->max_scale_value;
            local_min_scale_value = this->min_scale_value;
        }


        upsample_and_fit_scatter_estimate(*this->scaled_est_projdata_sptr, *this->input_projdata_2d_sptr,
                                          *unscaled, *empty,
                                          *this->mask_projdata_sptr, local_min_scale_value,
                                          local_max_scale_value, this->half_filter_width,
                                          spline_type, this->remove_interleaving);


        if(this->run_debug_mode)
        {
            std::stringstream convert;   // stream used for the conversion
            int lab = i_scat_iter+1;
            convert << "./extras/scaled_" << lab;
            std::string output_filename =  convert.str();

            dynamic_cast<ProjDataInMemory *> (this->scaled_est_projdata_sptr.get())->write_to_file(output_filename);
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

            // On the same time we are going to save to a temp projdata file

            shared_ptr<ProjData> temp_projdata ( new ProjDataInMemory (this->scaled_est_projdata_sptr->get_exam_info_sptr(),
                                                                       this->scaled_est_projdata_sptr->get_proj_data_info_sptr()));
            temp_projdata->fill(*this->scaled_est_projdata_sptr);

            pow_times_add min_threshold (0.0f, 1.0f, 1.0f, 1e-9f, NumericInfo<float>().max_value());

            for (int segment_num = temp_projdata->get_min_segment_num();
                 segment_num <= temp_projdata->get_max_segment_num();
                 ++segment_num)
            {
                SegmentByView<float> segment_by_view =
                        temp_projdata->get_segment_by_view(segment_num);

                in_place_apply_function(segment_by_view,
                                        min_threshold);

                if (!(temp_projdata->set_segment(segment_by_view) == Succeeded::yes))
                    warning("Error set_segment %d\n", segment_num);
            }

            pow_times_add add_scalar(-1e-9f, 1.0f, 1.0f, NumericInfo<float>().min_value(),
                                     NumericInfo<float>().max_value());

            for (int segment_num = temp_projdata->get_min_segment_num();
                 segment_num <= temp_projdata->get_max_segment_num();
                 ++segment_num)
            {
                SegmentByView<float> segment_by_view =
                        temp_projdata->get_segment_by_view(segment_num);

                in_place_apply_function(segment_by_view,
                                        add_scalar);

                if (!(temp_projdata->set_segment(segment_by_view) == Succeeded::yes))
                    warning("Error set_segment %d\n", segment_num);
            }

            // ok, we can multiply with the norm
            for (int segment_num = temp_projdata->get_min_segment_num();
                 segment_num <= temp_projdata->get_max_segment_num();
                 ++segment_num)
            {
                SegmentByView<float> segment_by_view_data =
                        temp_projdata->get_segment_by_view(segment_num);

                SegmentByView<float> segment_by_view_norm =
                        this->norm_projdata_2d_sptr->get_segment_by_view(segment_num);

                segment_by_view_data *= segment_by_view_norm;


                if (!(temp_projdata->set_segment(segment_by_view_data) == Succeeded::yes))
                    warning("Error set_segment %d\n", segment_num);
            }

            std::stringstream convert;   // stream used for the conversion
            convert << this->o_scatter_estimate_prefix << "_" <<
                       i_scat_iter+1;
            std::string output_filename = convert.str();

            BinNormalisationFromProjData normalisation_coeffs(this->norm_projdata_sptr);

            shared_ptr <ProjData> temp_projdata_3d (
                        new ProjDataInterfile(this->input_projdata_sptr->get_exam_info_sptr(),
                                              this->input_projdata_sptr->get_proj_data_info_sptr() ,
                                              output_filename,
                                              std::ios::in | std::ios::out | std::ios::trunc));
            // Upsample to 3D
            upsample_and_fit_scatter_estimate(*temp_projdata_3d,
                                              *this->input_projdata_sptr,
                                              *temp_projdata,
                                              normalisation_coeffs,
                                              *this->input_projdata_sptr,
                                              1.0f, 1.0f, 1.0f, spline_type,
                                              false);

            /*const double start_time = 0.0;
                                const double end_time = 1.0;
                                dynamic_cast<ChainedBinNormalisation &>
                                        (*this->multiplicative_data_3d).undo(*temp_projdata_3d,
                                                                             start_time,
                                                                             end_time);*/

            info ("\n\n Multiplying by normalisation factors \n\n");
            for (int segment_num = temp_projdata_3d->get_min_segment_num();
                 segment_num <= temp_projdata_3d->get_max_segment_num();
                 ++segment_num)
            {
                SegmentByView<float> segment_by_view_data =
                        temp_projdata_3d->get_segment_by_view(segment_num);

                SegmentByView<float> segment_by_view_norm =
                        this->norm_projdata_sptr->get_segment_by_view(segment_num);

                SegmentByView<float> segment_by_view_atten =
                        this->atten_projdata_sptr->get_segment_by_view(segment_num);

                SegmentByView<float> segment_by_view_back =
                        this->back_projdata_sptr->get_segment_by_view(segment_num);

                segment_by_view_data += segment_by_view_back;
                segment_by_view_data *= segment_by_view_norm * segment_by_view_atten;

                if (!(temp_projdata_3d->set_segment(segment_by_view_data) == Succeeded::yes))
                {
                    warning("Error set_segment %d\n", segment_num);
                    return Succeeded::no;
                }
            }


        }

        if (iterative_method)
            reconstruct_iterative(i_scat_iter,
                                  this->activity_image_lowres_sptr);
        else
        {
            reconstruct_analytic();
            //TODO: threshold ... to cut the negative values
        }

        // Reset to the additive factor
        //                this->scaled_est_projdata_sptr->fill(*this->back_projdata_sptr);

    }

    info("\n\n>>>>>>Scatter Estimation finished !!!<<<<<<<<<\n\n");

    return Succeeded::yes;
}


Succeeded
ScatterEstimationByBin::
reconstruct_iterative(int _current_iter_num,
                      shared_ptr<DiscretisedDensity<3, float> > & _current_estimate_sptr)
{

    IterativeReconstruction <DiscretisedDensity<3, float> > * iterative_object =
            dynamic_cast<IterativeReconstruction<DiscretisedDensity<3, float> > *> (this->reconstruction_template_sptr.get());

    iterative_object->set_start_subset_num(1);
    //    return iterative_object->reconstruct(this->activity_image_lowres_sptr);
    iterative_object->reconstruct(this->activity_image_lowres_sptr);

    if(this->run_debug_mode)
    {
        std::stringstream name;
        int next_iter_num = _current_iter_num+1;
        name << "./extras/recon_" << next_iter_num;
        std::string output = name.str();
        OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                write_to_file(output, *_current_estimate_sptr);
    }

}

Succeeded
ScatterEstimationByBin::
reconstruct_analytic()
{
    //TODO
}

/****************** functions to help **********************/

void
ScatterEstimationByBin::
write_log(const double simulation_time,
          const float total_scatter)
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

Succeeded
ScatterEstimationByBin::ffw_project_mask_image()
{
    if (is_null_ptr(this->mask_atten_image_sptr))
    {
        warning("You cannot forward project if you have not set the mask image. Abord.");
        return Succeeded::no;
    }

    if (is_null_ptr(this->input_projdata_2d_sptr))
    {
        warning("No 2D proj_data have been initialised. Abord.");
        return Succeeded::no;
    }

    shared_ptr<ForwardProjectorByBin> forw_projector_sptr;
    shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
    forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    info(boost::format("\n\nForward projector used for the calculation of\n"
                       "attenuation coefficients: %1%\n") % forw_projector_sptr->parameter_info());

    forw_projector_sptr->set_up(this->input_projdata_2d_sptr->get_proj_data_info_ptr()->create_shared_clone(),
                                this->mask_atten_image_sptr );

    shared_ptr<ProjData> mask_projdata(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
                                                            this->input_projdata_2d_sptr->get_proj_data_info_ptr()->create_shared_clone()));

    forw_projector_sptr->forward_project(*mask_projdata, *this->mask_atten_image_sptr);

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
            warning("Error set_segment %d\n", segment_num);
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
        warning(boost::format("Error parsing parameters file %1%, for creating mask tails from ACFs. Abord.")
                %this->tail_mask_par_filename);
        return Succeeded::no;
    }

    create_tail_mask_from_acfs.set_input_projdata_sptr(mask_projdata);
    create_tail_mask_from_acfs.set_output_projdata_sptr(this->mask_projdata_sptr);
    return create_tail_mask_from_acfs.process_data();
}

bool
ScatterEstimationByBin::
apply_mask_in_place(shared_ptr<DiscretisedDensity<3, float> >& arg,
                    const mask_parameters& _this_mask)
{
    // Re-reading every time should not be a problem, as it is
    // just a small txt file.
    PostFiltering<DiscretisedDensity<3, float> > filter;

    if(filter.parse(this->mask_postfilter_filename.c_str()) == false)
    {
        warning(boost::format("Error parsing postfilter parameters file %1%. Abord.")
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
    filter.process_data(*arg.get());

    // 2. max threshold
    in_place_apply_function(*arg.get(),
                            pow_times_thres_max);
    // 3. add scalar
    in_place_apply_function(*arg.get(),
                            pow_times_add_scalar);
    // 4. min threshold
    in_place_apply_function(*arg.get(),
                            pow_times_thres_min);
    // 5. times scalar
    in_place_apply_function(*arg.get(),
                            pow_times_times);
    // 6. Add 1.
    in_place_apply_function(*arg.get(),
                            pow_times_add_object);

    return true;
}

END_NAMESPACE_STIR
