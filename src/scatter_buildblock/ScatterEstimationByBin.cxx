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
#include "stir/PostFiltering.h"
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

#include "stir/recon_buildblock/IterativeReconstruction.h"

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
    this->half_filter_width = 3.f;
    this->iterative_method = true;
    this->do_average_at_2 = true;
    this->export_scatter_estimates_of_each_iteration = false;
    this->run_debug_mode = false;
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
                         &this->normalisation_projdata_filename);

    this->parser.add_key("recompute initial activity image",
                         &this->recompute_initial_activity_image);
    this->parser.add_key("initial activity image filename",
                         &this->initial_activity_image_filename);

    // ITERATIONS RELATED
    this->parser.add_key("reconstruction template filename",
                         &this->reconstruction_template_par_filename);
    this->parser.add_key("number of scatter iterations",
                         &this->num_scatter_iterations);
    //END ITERATIONS RELATED

    //Scatter simulation
    this->parser.add_key("scatter simulation parameters",
                         &this->scatter_sim_par_filename);

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

bool
ScatterEstimationByBin::
post_processing()
{
    if (this->run_debug_mode)
    {
        info("Debugging mode is activated.");
        this->export_2d_projdata = true;
        this->export_scatter_estimates_of_each_iteration = true;
    }

    info("Loading input projection data");
    this->input_projdata_3d_sptr =
            ProjData::read_from_file(this->input_projdata_filename);

    // Load InputProjData and calculate the SSRB
    if (this->input_projdata_3d_sptr->get_max_segment_num() > 0 )
    {
        this->proj_data_info_2d_sptr.reset(
                    dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >
                    (SSRB(*this->input_projdata_3d_sptr->get_proj_data_info_ptr(),
                          this->input_projdata_3d_sptr->get_max_segment_num())));

        if (this->export_2d_projdata)
        {
            size_t lastindex = this->input_projdata_filename.find_last_of(".");
            std::string rawname = this->input_projdata_filename.substr(0, lastindex);
            std::string out_filename = rawname + "_2d.hs";

            this->input_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_3d_sptr->get_exam_info_sptr(),
                                                                     this->proj_data_info_2d_sptr,
                                                                     out_filename,
                                                                     std::ios::in | std::ios::out | std::ios::trunc));
        }
        else
            this->input_projdata_2d_sptr.reset(new ProjDataInMemory(this->input_projdata_3d_sptr->get_exam_info_sptr(),
                                                                    this->proj_data_info_2d_sptr));

        SSRB(*this->input_projdata_2d_sptr,
             *input_projdata_3d_sptr,false);
    }
    else
    {
        error(boost::format("The input data %1% are not 3D") % this->input_projdata_filename);
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
        info("Initialising initial activity image ... ");
        this->activity_image_sptr.reset(new VoxelsOnCartesianGrid<float> ( *this->input_projdata_2d_sptr->get_proj_data_info_sptr() ) );
        this->activity_image_sptr->fill(1.F);
    }

    //
    // Initialise the reconstruction method and output image
    // We should not run set_up() at this point, because first we should add
    // normalisation data.
    //

    if (this->reconstruction_template_par_filename.size() > 0 )
    {
        KeyParser local_parser;
        local_parser.add_start_key("Reconstruction");
        local_parser.add_stop_key("End Reconstruction");
        local_parser.add_parsing_key("reconstruction method", &this->reconstruction_template_sptr);
        local_parser.parse(this->reconstruction_template_par_filename.c_str());
    }
    else
    {
        error("Please define a reconstruction method.");
    }

    //
    // Load the attenuation image.
    //
    {
        info("Loading attenuation image...");
        this->atten_image_sptr =
                read_from_file<DiscretisedDensity<3,float> >(this->atten_image_filename);

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
    //
    info("Zooming activity and attenuation images...");

    VoxelsOnCartesianGrid<float>* activity_image_ptr =
            dynamic_cast<VoxelsOnCartesianGrid<float>* >(this->activity_image_sptr.get());
    VoxelsOnCartesianGrid<float>* attenuation_image_ptr =
            dynamic_cast<VoxelsOnCartesianGrid<float>* >(this->atten_image_sptr.get());

    // zoom image
    float zoom_z = 1.0f;
    float zoom_xy = 0.3f;

    int size_xy = activity_image_ptr->get_x_size() * zoom_xy;
    int size_z = activity_image_ptr->get_z_size() * zoom_z;

    this->activity_image_lowres_sptr.reset( new VoxelsOnCartesianGrid<float>());

    VoxelsOnCartesianGrid<float> activity_lowres =
            zoom_image(*activity_image_ptr,
                       CartesianCoordinate3D<float>(zoom_z, zoom_xy, zoom_xy),
                       CartesianCoordinate3D<float>(0.0f, 0.0f, 0.0f),
                       CartesianCoordinate3D<int>(size_z, size_xy, size_xy));

    this->activity_image_lowres_sptr.reset(activity_lowres.clone());
    this->activity_image_lowres_sptr->fill(1.f);


    // zoom the attenuation image to the same size;
    info("Zooming attenuation image to much the activity image");
    shared_ptr< DiscretisedDensity<3, float> > attenuation_lowres(this->activity_image_lowres_sptr->get_empty_copy());
    VoxelsOnCartesianGrid<float>* attenuation_lowres_ptr =
            dynamic_cast<VoxelsOnCartesianGrid<float>* >(attenuation_lowres.get());

    zoom_image(*attenuation_lowres_ptr, *attenuation_image_ptr);

    PostFiltering filter;
    filter.parser.parse(this->mask_postfilter_filename.c_str());
    filter.filter_ptr->apply(*attenuation_lowres_ptr);

    float scale_att = zoom_xy * zoom_xy * zoom_z;

    // Just multiply with the scale factor.
    pow_times_add scale_factor(0.0f, scale_att, 1.0f,
                               NumericInfo<float>().min_value(),
                               NumericInfo<float>().max_value());

    in_place_apply_function(*attenuation_lowres_ptr, scale_factor);

    this->atten_image_lowres_sptr.reset(attenuation_lowres->clone());

    info("Initialising reconstruction ...");
    if (this->reconstruction_template_sptr->get_registered_name() == "OSMAPOSL" ||
            this->reconstruction_template_sptr->get_registered_name() == "OSSPS")
    {
        if (set_up_iterative() == false)
            error("Initialisation Failed!");

        this->iterative_method = true;
    }
    else if (this->reconstruction_template_sptr->get_registered_name() == "FBP3D" ||
             this->reconstruction_template_sptr->get_registered_name() == "FBP2D")
    {
        if (set_up_analytic() == false)
            error("Initialisation Failed!");

        this->iterative_method = false;
    }

    //
    // (only after this->set_up[...]
    // We can call set up reconstuction.
    info("Setting up reconstruction ...");
//    this->reconstruction_template_sptr->set_up(this->activity_image_lowres_sptr);

    //
    // ScatterSimulation
    //

    info ("Initialising Scatter Simulation ... ");
    if (this->scatter_sim_par_filename.size() > 0 )
    {
        KeyParser local_parser;
        local_parser.add_start_key("Scatter Simulation");
        local_parser.add_stop_key("End Scatter Simulation");
        local_parser.add_parsing_key("Simulation method", &this->scatter_simulation_sptr);
        local_parser.parse(this->scatter_sim_par_filename.c_str());

        // The image is provided to the simulation.
        // and it will override anything that the ScatterSimulation.par file has done.
        this->scatter_simulation_sptr->set_density_image_sptr(this->atten_image_lowres_sptr);
        this->scatter_simulation_sptr->set_density_image_for_scatter_points_sptr(this->atten_image_lowres_sptr);
        this->scatter_simulation_sptr->set_activity_image_sptr(this->activity_image_lowres_sptr);

    }
    else
    {
        error("Please define a scatter simulation method.");
    }

    //
    // Initialise the mask image.
    //
    if (this->recompute_mask_atten_image)
    {
        // Applying mask
        // 1. Clone from the original image.
        // 2. Apply to the new clone.
        this->mask_atten_image_sptr.reset(this->atten_image_sptr->clone());
        this->apply_mask_in_place(this->mask_atten_image_sptr,
                                  this->mask_attenuation_image);

        if (this->mask_atten_image_filename.size() > 0 )
            OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                    write_to_file(this->mask_atten_image_filename, *this->mask_atten_image_sptr.get());
    }
    else if (!this->recompute_mask_atten_image && this->mask_atten_image_filename.size() > 0)
    {
        this->mask_atten_image_sptr =
                read_from_file<DiscretisedDensity<3, float> >(this->mask_atten_image_filename);
    }
    else
        error ("Please set the postfilter parameter filename or set to recompute it.");

    // Forward Project the mask
    if (this->recompute_mask_projdata)
    {
        ffw_project_mask_image();
    }
    else // Load from file
    {
        if (this->mask_projdata_filename.size() > 0)
            this->mask_projdata_sptr =
                ProjData::read_from_file(this->mask_projdata_filename);
        else
            error (boost::format("Mask projdata file %1% not found") %this->mask_projdata_filename );
    }

    return false;
}

ScatterEstimationByBin::
ScatterEstimationByBin()
{
    this->set_defaults();
}

bool
ScatterEstimationByBin::
set_up_iterative()
{
    //    IterativeReconstruction <DiscretisedDensity<3, float> > * iterative_object =
    //            dynamic_cast<IterativeReconstruction<DiscretisedDensity<3, float> > *> (this->reconstruction_template_sptr.get());

    //    this->reconstruction_template_sptr->set_input_data(this->input_projdata_2d_sptr);

    //    //
    //    // Multiplicative projdata
    //    //

    //    shared_ptr<BinNormalisation> attenuation_correction(new TrivialBinNormalisation());
    //    shared_ptr<BinNormalisation> normalisation_coeffs(new TrivialBinNormalisation());

    //    shared_ptr<BinNormalisation> attenuation_correction_3d(new TrivialBinNormalisation());
    //    shared_ptr<BinNormalisation> normalisation_coeffs_3d(new TrivialBinNormalisation());

    //    // Attenuation projdata
    //    if (!this->recompute_atten_projdata)
    //    {
    //        if  (this->atten_coeff_filename.size() == 0)
    //            error("No attenuation projdata filename is given. Abort ");

    //        // Read ProjData and make them 2D

    //        this->atten_projdata_3d_sptr =
    //                ProjData::read_from_file(this->atten_coeff_filename);
    //    }
    //    else
    //    {
    //        //Project the attenuation imge.
    //        shared_ptr<ForwardProjectorByBin> forw_projector_sptr;
    //        shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
    //        forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    //        info(boost::format("\n\nForward projector used for the calculation of\n"
    //                           "attenuation coefficients: %1%\n") % forw_projector_sptr->parameter_info());

    //        forw_projector_sptr->set_up(this->input_projdata_3d_sptr->get_proj_data_info_ptr()->create_shared_clone(),
    //                                    this->atten_image_sptr );

    //        // I think I could projec to 2D ...
    //        // - No, I would still need the 3D version at the very end.
    //        if (this->atten_coeff_filename.size() > 0)
    //            this->atten_projdata_3d_sptr.reset(new ProjDataInterfile(this->input_projdata_3d_sptr->get_exam_info_sptr(),
    //                                                                     this->input_projdata_3d_sptr->get_proj_data_info_sptr()->create_shared_clone(),
    //                                                                     this->atten_coeff_filename,
    //                                                                     std::ios::in | std::ios::out | std::ios::trunc));
    //        else
    //            this->atten_projdata_3d_sptr.reset(new ProjDataInMemory(this->input_projdata_3d_sptr->get_exam_info_sptr(),
    //                                                                    this->input_projdata_3d_sptr->get_proj_data_info_ptr()->create_shared_clone()));

    //        forw_projector_sptr->forward_project(* this->atten_projdata_3d_sptr, *this->atten_image_sptr);
    //    }


    //    if( this->atten_projdata_3d_sptr->get_max_segment_num() > 0)
    //    {

    //        if(this->export_2d_projdata)
    //        {
    //            size_t lastindex = this->atten_coeff_filename.find_last_of(".");
    //            std::string rawname = this->atten_coeff_filename.substr(0, lastindex);
    //            std::string out_filename = rawname + "_2d.hs";

    //            this->atten_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
    //                                                                     this->proj_data_info_2d_sptr,
    //                                                                     out_filename,
    //                                                                     std::ios::in | std::ios::out | std::ios::trunc));
    //        }
    //        else
    //            this->atten_projdata_2d_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
    //                                                                    this->proj_data_info_2d_sptr));


    //        SSRB(*this->atten_projdata_2d_sptr,
    //             *this->atten_projdata_3d_sptr, true);
    //    }
    //    else
    //    {
    //        error(boost::format("The attenuation projdata %1% are not 3D") % this->atten_coeff_filename);
    //    }

    //    attenuation_correction.reset(new BinNormalisationFromProjData(this->atten_projdata_2d_sptr));
    //    attenuation_correction_3d.reset(new BinNormalisationFromProjData(this->atten_projdata_3d_sptr));

    //    //    else if (this->atten_coeff_filename.size() == 0)
    //    //    {
    //    //        attenuation_correction.reset(new BinNormalisationFromAttenuationImage(this->atten_image_lowres_sptr));
    //    //    }
    //    //<- End of Attenuation projdata

    //    // Normalisation ProjData
    //    if (this->normalisation_projdata_filename.size() > 0 )
    //    {
    //        // Read ProjData and make them 2D

    //        this->norm_projdata_3d_sptr =
    //                ProjData::read_from_file(this->normalisation_projdata_filename);

    //        if( this->norm_projdata_3d_sptr->get_max_segment_num() > 0) //If the sinogram is 3D then process it.
    //        {

    //            shared_ptr < ProjData> inv_norm_projdata_sptr(new ProjDataInMemory(this->norm_projdata_3d_sptr->get_exam_info_sptr(),
    //                                                                               this->norm_projdata_3d_sptr->get_proj_data_info_ptr()->create_shared_clone()));

    //            //            inv_norm_projdata_sptr->fill(*this->norm_projdata_3d_sptr);
    //            inv_norm_projdata_sptr->fill(0.f);
    //            // We need to get norm2d=1/SSRB(1/norm3d))
    //            //1. add_scalar
    //            //2. mult_scalar
    //            //3. power
    //            //4. min_threshold
    //            //5. max_threshold

    //            pow_times_add pow_times_add_object(0.0f, 1.0f, -1.0f, NumericInfo<float>().min_value(),
    //                                               NumericInfo<float>().max_value());

    //            for (int segment_num = inv_norm_projdata_sptr->get_min_segment_num();
    //                 segment_num <= inv_norm_projdata_sptr->get_max_segment_num();
    //                 ++segment_num)
    //            {
    //                SegmentByView<float> segment_by_view =
    //                        this->norm_projdata_3d_sptr->get_segment_by_view(segment_num);

    //                in_place_apply_function(segment_by_view,
    //                                        pow_times_add_object);

    //                if (!(inv_norm_projdata_sptr->set_segment(segment_by_view) == Succeeded::yes))
    //                    warning("Error set_segment %d\n", segment_num);
    //            }

    //            if (this->export_2d_projdata)
    //            {
    //                size_t lastindex = this->normalisation_projdata_filename.find_last_of(".");
    //                std::string rawname = this->normalisation_projdata_filename.substr(0, lastindex);
    //                std::string out_filename = rawname + "_2d.hs";

    //                this->norm_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
    //                                                                        this->proj_data_info_2d_sptr,
    //                                                                        out_filename,
    //                                                                        std::ios::in | std::ios::out | std::ios::trunc));
    //            }
    //            else
    //                this->norm_projdata_2d_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
    //                                                                       this->proj_data_info_2d_sptr));

    //            SSRB(*this->norm_projdata_2d_sptr,
    //                 *inv_norm_projdata_sptr,false);

    //            for (int segment_num = this->norm_projdata_2d_sptr->get_min_segment_num();
    //                 segment_num <= this->norm_projdata_2d_sptr->get_max_segment_num();
    //                 ++segment_num)
    //            {
    //                SegmentByView<float> segment_by_view =
    //                        this->norm_projdata_2d_sptr->get_segment_by_view(segment_num);

    //                in_place_apply_function(segment_by_view,
    //                                        pow_times_add_object);

    //                if (!(this->norm_projdata_2d_sptr->set_segment(segment_by_view) == Succeeded::yes))
    //                    warning("Error set_segment %d\n", segment_num);
    //            }


    //        }
    //        else
    //        {
    //            error(boost::format("The normalisation projdata %1% are not 3D") % this->normalisation_projdata_filename);
    //        }

    //        normalisation_coeffs.reset(new BinNormalisationFromProjData(this->norm_projdata_2d_sptr));
    //        normalisation_coeffs_3d.reset(new BinNormalisationFromProjData(this->norm_projdata_3d_sptr));
    //    }
    //    //<- End Normalisation ProjData

    //    // Multiply atten with norm


    //    this->multimulti2d.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
    //                                                  this->proj_data_info_2d_sptr));

    //    // ok, we can multiply with the norm
    //    for (int segment_num = this->atten_projdata_2d_sptr->get_min_segment_num();
    //         segment_num <= this->atten_projdata_2d_sptr->get_max_segment_num();
    //         ++segment_num)
    //    {
    //        SegmentByView<float> segment_by_view_data =
    //                this->atten_projdata_2d_sptr->get_segment_by_view(segment_num);

    //        SegmentByView<float> segment_by_view_norm =
    //                this->norm_projdata_2d_sptr->get_segment_by_view(segment_num);

    //        segment_by_view_data *= segment_by_view_norm;


    //        if (!(this->multimulti2d->set_segment(segment_by_view_data) == Succeeded::yes))
    //            warning("Error set_segment %d\n", segment_num);
    //    }


    //    this->multiplicative_data_2d.reset(new ChainedBinNormalisation(attenuation_correction,
    //                                                                   normalisation_coeffs));

    //    this->multiplicative_data_3d.reset(new ChainedBinNormalisation(attenuation_correction_3d,
    //                                                                   normalisation_coeffs_3d));

    //    this->normalisation_data_3d.reset(new BinNormalisationFromProjData(this->norm_projdata_3d_sptr));
    //    //    this->multiplicative_data_3d.reset(new BinNormalisationFromProjData(this->norm_projdata_3d_sptr));
    //    //    this->multiplicative_data_3d.reset(new TrivialBinNormalisation());

    //    this->only_atten.reset(new BinNormalisationFromProjData(this->atten_projdata_2d_sptr));

    //    iterative_object->set_normalisation_sptr(this->multiplicative_data_2d);

    //    //
    //    // Set additive (background) projdata
    //    //

    //    if (this->back_projdata_filename.size() > 0)
    //    {
    //        shared_ptr<ProjData> back_projdata_sptr =
    //                ProjData::read_from_file(this->back_projdata_filename);

    //        if( back_projdata_sptr->get_max_segment_num() > 0)
    //        {
    //            size_t lastindex = this->back_projdata_filename.find_last_of(".");
    //            std::string rawname = this->back_projdata_filename.substr(0, lastindex);
    //            std::string out_filename = rawname + "_2d.hs";

    //            this->back_projdata_2d_sptr.reset(new ProjDataInterfile(this->input_projdata_2d_sptr->get_exam_info_sptr(),
    //                                                                    this->proj_data_info_2d_sptr,
    //                                                                    out_filename,
    //                                                                    std::ios::in | std::ios::out | std::ios::trunc));

    //            SSRB(*this->back_projdata_2d_sptr,
    //                 *back_projdata_sptr.get(), false);
    //        }
    //        else
    //        {
    //            this->back_projdata_2d_sptr = back_projdata_sptr;
    //        }

    //        iterative_object->set_additive_proj_data_sptr(this->back_projdata_2d_sptr);
    //    }

    //    // Allocate the output projdata.
    //    this->output_projdata_sptr.reset(new ProjDataInMemory(this->input_projdata_2d_sptr->get_exam_info_sptr(),
    //                                                          this->input_projdata_2d_sptr->get_proj_data_info_sptr()->create_shared_clone()));

    //    this->output_projdata_sptr->fill(0.F);

    //    iterative_object->set_additive_proj_data_sptr(this->output_projdata_sptr);

    return true;
}

bool
ScatterEstimationByBin::
set_up_analytic()
{
    //TODO : I have most stuff in tmp.
    return true;
}

Succeeded
ScatterEstimationByBin::
process_data()
{

    //    float local_min_scale_value = 0.5f;
    //    float local_max_scale_value = 0.5f;

    //    stir::BSpline::BSplineType  spline_type = stir::BSpline::linear;
    //    shared_ptr <ProjData> unscaled;
    //    shared_ptr<BinNormalisation> empty(new TrivialBinNormalisation());
    //    shared_ptr<DiscretisedDensity <3,float> > act_image_for_averaging;

    //    this->scatter_simulation_sptr->get_output_proj_data(unscaled);

    //    if( this->recompute_initial_activity_image )
    //    {
    //        if (iterative_method)
    //            reconstruct_iterative( 0,
    //                                   this->activity_image_lowres_sptr);
    //        else
    //            reconstruct_analytic();

    //        if (this->initial_activity_image_filename.size() > 0 )
    //            OutputFileFormat<DiscretisedDensity < 3, float > >::default_sptr()->
    //                    write_to_file(this->initial_activity_image_filename, *this->activity_image_lowres_sptr);
    //    }
    //    else // load
    //    {
    //        this->activity_image_lowres_sptr =
    //                read_from_file<DiscretisedDensity<3,float> >(this->initial_activity_image_filename);
    //    }

    //    if ( this->do_average_at_2)
    //        act_image_for_averaging.reset(this->activity_image_lowres_sptr->clone());

    //    //
    //    // Begin the estimation process...
    //    //
    //    for (int i_scat_iter = 0;
    //         i_scat_iter < this->num_scatter_iterations;
    //         i_scat_iter++)
    //    {


    //        if ( this->do_average_at_2)
    //        {
    //            if (i_scat_iter == 2) // do average 0 and 1
    //            {
    //                if (is_null_ptr(act_image_for_averaging))
    //                    error("Storing the first actibity estimate has failed at some point.");

    //                if (this->run_debug_mode)// Write the previous estimate
    //                {
    //                    std::string output1= "./extras/act_image_for_averaging";
    //                    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    //                            write_to_file(output1, *act_image_for_averaging);
    //                }
    //                std::transform(act_image_for_averaging->begin_all(), act_image_for_averaging->end_all(),
    //                               this->activity_image_lowres_sptr->begin_all(),
    //                               this->activity_image_lowres_sptr->begin_all(),
    //                               std::plus<float>());

    //                if(this->run_debug_mode)// write after sum
    //                {
    //                    std::string output1= "./extras/summed_for_averaging";
    //                    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    //                            write_to_file(output1, *this->activity_image_lowres_sptr);
    //                }

    //                pow_times_add divide_by_two (0.0f, 0.5f, 1.0f, NumericInfo<float>().min_value(),
    //                                             NumericInfo<float>().max_value());

    //                in_place_apply_function(*this->activity_image_lowres_sptr, divide_by_two);

    //                if(this->run_debug_mode)// write after division.
    //                {
    //                    std::string output1= "./extras/recon_0_1";
    //                    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    //                            write_to_file(output1, *this->activity_image_lowres_sptr);
    //                }

    //                // Have averaged ... turn false
    //                this->do_average_at_2 = false;
    //            }
    //        }

    //        //filter ... if FBP

    //        info("Scatter simulation in progress...");
    //        this->scatter_simulation_sptr->set_activity_image_sptr(this->activity_image_lowres_sptr);
    //        this->scatter_simulation_sptr->process_data();
    //        this->scatter_simulation_sptr->get_output_proj_data(unscaled);

    //        if(this->run_debug_mode) // Write unscaled scatter sinogram
    //        {
    //            std::stringstream convert;   // stream used for the conversion
    //            int lab = i_scat_iter+1;
    //            convert << "./extras/unscaled_" <<
    //                       lab;
    //            std::string output_filename =  convert.str();

    //            dynamic_cast<ProjDataInMemory *> (unscaled.get())->write_to_file(output_filename);
    //        }

    //        // end simulation

    //        // Set the min and max scale factors
    //        if (i_scat_iter > 0)
    //        {
    //            local_max_scale_value = this->max_scale_value;
    //            local_min_scale_value = this->min_scale_value;
    //        }


    //        upsample_and_fit_scatter_estimate(*this->output_projdata_sptr,
    //                                          *this->input_projdata_2d_sptr,
    //                                          *unscaled,
    //                                          *empty,
    //                                          *this->mask_projdata_sptr,
    //                                          local_min_scale_value,
    //                                          local_max_scale_value,
    //                                          this->half_filter_width,
    //                                          spline_type,
    //                                          this->remove_interleaving);


    //        if(this->run_debug_mode)
    //        {
    //            std::stringstream convert;   // stream used for the conversion
    //            int lab = i_scat_iter+1;
    //            convert << "./extras/scaled_" <<
    //                       lab;
    //            std::string output_filename =  convert.str();

    //            dynamic_cast<ProjDataInMemory *> (this->output_projdata_sptr.get())->write_to_file(output_filename);
    //        }

    //        if (this->export_scatter_estimates_of_each_iteration ||
    //                i_scat_iter == this->num_scatter_iterations -1 )
    //        {

    //            //this is complicated as the 2d scatter estimate was
    //            //divided by norm2d, so we need to undo this
    //            //unfortunately, currently the values in the gaps in the
    //            //scatter estimate are not quite zero (just very small)
    //            //so we have to first make sure that they are zero before
    //            //we do any of this, otherwise the values after normalisation will be garbage
    //            //we do this by min-thresholding and then subtracting the threshold.
    //            //as long as the threshold is tiny, this will be ok

    //            // On the same time we are going to save to a temp projdata file

    //            shared_ptr<ProjData> temp_projdata ( new ProjDataInMemory (this->output_projdata_sptr->get_exam_info_sptr(),
    //                                                                       this->output_projdata_sptr->get_proj_data_info_sptr()));
    //            temp_projdata->fill(*this->output_projdata_sptr);

    //            pow_times_add min_threshold (0.0f, 1.0f, 1.0f, 1e-9, NumericInfo<float>().max_value());

    //            for (int segment_num = temp_projdata->get_min_segment_num();
    //                 segment_num <= temp_projdata->get_max_segment_num();
    //                 ++segment_num)
    //            {
    //                SegmentByView<float> segment_by_view =
    //                        temp_projdata->get_segment_by_view(segment_num);

    //                in_place_apply_function(segment_by_view,
    //                                        min_threshold);

    //                if (!(temp_projdata->set_segment(segment_by_view) == Succeeded::yes))
    //                    warning("Error set_segment %d\n", segment_num);
    //            }

    //            pow_times_add add_scalar(-1e-9, 1.0f, 1.0f, NumericInfo<float>().min_value(),
    //                                     NumericInfo<float>().max_value());

    //            for (int segment_num = temp_projdata->get_min_segment_num();
    //                 segment_num <= temp_projdata->get_max_segment_num();
    //                 ++segment_num)
    //            {
    //                SegmentByView<float> segment_by_view =
    //                        temp_projdata->get_segment_by_view(segment_num);

    //                in_place_apply_function(segment_by_view,
    //                                        add_scalar);

    //                if (!(temp_projdata->set_segment(segment_by_view) == Succeeded::yes))
    //                    warning("Error set_segment %d\n", segment_num);
    //            }

    //            // ok, we can multiply with the norm
    //            for (int segment_num = temp_projdata->get_min_segment_num();
    //                 segment_num <= temp_projdata->get_max_segment_num();
    //                 ++segment_num)
    //            {
    //                SegmentByView<float> segment_by_view_data =
    //                        temp_projdata->get_segment_by_view(segment_num);

    //                SegmentByView<float> segment_by_view_norm =
    //                        this->norm_projdata_2d_sptr->get_segment_by_view(segment_num);

    //                segment_by_view_data *= segment_by_view_norm;


    //                if (!(temp_projdata->set_segment(segment_by_view_data) == Succeeded::yes))
    //                    warning("Error set_segment %d\n", segment_num);
    //            }

    //            std::stringstream convert;   // stream used for the conversion
    //            convert << this->o_scatter_estimate_prefix << "_" <<
    //                       i_scat_iter;
    //            std::string output_filename =  convert.str();

    //            shared_ptr <ProjData> temp_projdata_3d (new ProjDataInterfile(this->input_projdata_3d_sptr->get_exam_info_sptr(),
    //                                                                          this->input_projdata_3d_sptr->get_proj_data_info_sptr() ,
    //                                                                          output_filename,
    //                                                                          std::ios::in | std::ios::out | std::ios::trunc));

    //            upsample_and_fit_scatter_estimate(*temp_projdata_3d,
    //                                              *this->input_projdata_3d_sptr,
    //                                              *temp_projdata,
    //                                              *this->normalisation_data_3d,
    //                                              *this->input_projdata_3d_sptr,
    //                                              1.0f,
    //                                              1.0f,
    //                                              1.0f,
    //                                              spline_type,
    //                                              false);

    //            /*const double start_time = 0.0;
    //                    const double end_time = 1.0;
    //                    dynamic_cast<ChainedBinNormalisation &>
    //                            (*this->multiplicative_data_3d).undo(*temp_projdata_3d,
    //                                                                 start_time,
    //                                                                 end_time);*/

    //            info ("\n\n Multiplying \n\n");
    //            for (int segment_num = temp_projdata_3d->get_min_segment_num();
    //                 segment_num <= temp_projdata_3d->get_max_segment_num();
    //                 ++segment_num)
    //            {
    //                SegmentByView<float> segment_by_view_data =
    //                        temp_projdata_3d->get_segment_by_view(segment_num);

    //                SegmentByView<float> segment_by_view_norm =
    //                        this->norm_projdata_3d_sptr->get_segment_by_view(segment_num);

    //                SegmentByView<float> segment_by_view_atten =
    //                        this->atten_projdata_3d_sptr->get_segment_by_view(segment_num);

    //                segment_by_view_data *= segment_by_view_norm;
    //                segment_by_view_data *= segment_by_view_atten;


    //                if (!(temp_projdata_3d->set_segment(segment_by_view_data) == Succeeded::yes))
    //                    warning("Error set_segment %d\n", segment_num);
    //            }


    //        }

    //        if (iterative_method)
    //            reconstruct_iterative(i_scat_iter,
    //                                  this->activity_image_lowres_sptr);
    //        else
    //        {
    //            reconstruct_analytic();
    //            //TODO: threshold ... to cut the negative values
    //        }

    //        this->output_projdata_sptr->fill(0.f);

    //    }

    //    info("\n\n>>>>>>Scatter Estimation finished !!!<<<<<<<<<\n\n");

    //    // todo: Reconstruct in full 3D, with the last scatter estimation sinogram,
    //    // or with all.


    //    return Succeeded::yes;
}


Succeeded
ScatterEstimationByBin::
reconstruct_iterative(int _current_iter_num,
                      shared_ptr<DiscretisedDensity<3, float> > & _current_estimate_sptr)
{

    IterativeReconstruction <DiscretisedDensity<3, float> > * iterative_object =
            dynamic_cast<IterativeReconstruction<DiscretisedDensity<3, float> > *> (this->reconstruction_template_sptr.get());

    iterative_object->set_start_subset_num(1);
    iterative_object->reconstruct(_current_estimate_sptr);

    if(this->run_debug_mode)
    {
        std::stringstream name;
        int next_iter_num = _current_iter_num + 1;
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

void
ScatterEstimationByBin::ffw_project_mask_image()
{
    if (is_null_ptr(this->mask_atten_image_sptr))
        error("You cannot forward project if you have not set the mask image");

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
            warning("Error set_segment %d\n", segment_num);
    }

    if (this->mask_projdata_filename.size() > 0)
        this->mask_projdata_sptr.reset(new ProjDataInterfile(mask_projdata->get_exam_info_sptr(),
                                                             mask_projdata->get_proj_data_info_ptr()->create_shared_clone(),
                                                             this->mask_projdata_filename));
    else
        this->mask_projdata_sptr.reset(new ProjDataInMemory(mask_projdata->get_exam_info_sptr(),
                                                            mask_projdata->get_proj_data_info_ptr()->create_shared_clone()));

    CreateTailMaskFromACFs create_tail_mask_from_acfs;
    create_tail_mask_from_acfs.parse(this->tail_mask_par_filename.c_str());

    create_tail_mask_from_acfs.set_input_projdata_sptr(mask_projdata);
    create_tail_mask_from_acfs.set_output_projdata_sptr(this->mask_projdata_sptr);
    create_tail_mask_from_acfs.process_data();
}

void
ScatterEstimationByBin::
apply_mask_in_place(shared_ptr<DiscretisedDensity<3, float> >& arg,
                    const mask_parameters& _this_mask)
{
    // Re-reading every time should not be a problem, as it is
    // just a small txt file.
    PostFiltering filter;
    filter.parser.parse(this->mask_postfilter_filename.c_str());

    // How to use:
    //  pow_times_add(const float add_scalar,
    //  const float mult_scalar, const float power,
    //  const float min_threshold, const float max_threshold)

    //1. add_scalar
    //2. mult_scalar
    //3. power
    //4. min_threshold
    //5. max_threshold

    // The power is hardcoded because in this context has no
    // use.

    pow_times_add pow_times_thres_max(0.0f, 1.0f, 1.0f, NumericInfo<float>().min_value(),
                                      0.001);
    pow_times_add pow_times_add_scalar( -0.00099f, 1.0f, 1.0f, NumericInfo<float>().min_value(),
                                        NumericInfo<float>().max_value());

    pow_times_add pow_times_thres_min(0.0, 1.0f, 1.0f, 0.0f,
                                      NumericInfo<float>().max_value());

    pow_times_add pow_times_times( 0.0f, 100002.0f, 1.0f, NumericInfo<float>().min_value(),
                                   NumericInfo<float>().max_value());

    pow_times_add pow_times_add_object(_this_mask.add_scalar,
                                       _this_mask.times_scalar,
                                       1.0,
                                       _this_mask.min_threshold,
                                       _this_mask.max_threshold);

    // 1. filter the image
    filter.filter_ptr->apply(*arg.get());

    // 2. max threshold
    // 3. add scalar
    // 4. min threshold
    // 5. times scalar

    in_place_apply_function(*arg.get(),
                            pow_times_thres_max);
    in_place_apply_function(*arg.get(),
                            pow_times_add_scalar);
    in_place_apply_function(*arg.get(),
                            pow_times_thres_min);
    in_place_apply_function(*arg.get(),
                            pow_times_times);
    in_place_apply_function(*arg.get(),
                            pow_times_add_object);
}

END_NAMESPACE_STIR
