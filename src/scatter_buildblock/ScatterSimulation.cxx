/*
    Copyright (C) 2004 - 2009 Hammersmith Imanet Ltd
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
  \brief Definition of class stir::ScatterEstimationByBin.

  \author Nikos Efthimiou
  \author Kris Thielemans
*/
#include "stir/scatter/ScatterSimulation.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Bin.h"

#include "stir/Viewgram.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/error.h"
#include <fstream>
#include <boost/format.hpp>

#include "stir/zoom.h"
#include "stir/SSRB.h"

#include "stir/stir_math.h"
#include "stir/zoom.h"
#include "stir/NumericInfo.h"

START_NAMESPACE_STIR

ScatterSimulation::
ScatterSimulation()
{
    this->set_defaults();
}

ScatterSimulation::
~ScatterSimulation()
{}

Succeeded
ScatterSimulation::
process_data()
{
    // this is usefull in the scatter estimation process.
    this->output_proj_data_sptr->fill(0.f);
    //show energy window information

    std::cerr << "number of energy windows:= "<<  this->template_exam_info_sptr->get_num_energy_windows() << '\n';

    if(this->template_exam_info_sptr->get_energy_window_pair().first!= -1 &&
         this->template_exam_info_sptr->get_energy_window_pair().second!= -1 )
    {
        std::cerr << "energy window pair :="<<" {"<<  this->template_exam_info_sptr->get_energy_window_pair().first  << ',' <<  this->template_exam_info_sptr->get_energy_window_pair().second <<"}\n";

    }


    for (int i = 1; i <= this->template_exam_info_sptr->get_num_energy_windows(); ++i)
    {
        std::cerr << "energy window lower level"<<"["<<i<<"] := "<< this->template_exam_info_sptr->get_low_energy_thres(i-1) << '\n';
        std::cerr << "energy window upper level"<<"["<<i<<"] := "<<  this->template_exam_info_sptr->get_high_energy_thres(i-1) << '\n';

        if (this->template_exam_info_sptr->get_low_energy_thres(i-1) == -1.F || this->template_exam_info_sptr->get_high_energy_thres(i-1) == -1.F)

            {
                std::cerr << "Not enough input arguments. The energy window thresholds have to be specified for all the energy windows.\n" << '\n';
                return Succeeded::no;
            }

    }



    info("ScatterSimulator: Running Scatter Simulation ...");
    info("ScatterSimulator: Initialising ...");
    // The activiy image might have been changed, during the estimation process.
    this->remove_cache_for_integrals_over_activity();
    this->remove_cache_for_integrals_over_attenuation();
    this->sample_scatter_points();
    this->initialise_cache_for_scattpoint_det_integrals_over_attenuation();
    this->initialise_cache_for_scattpoint_det_integrals_over_activity();

    ViewSegmentNumbers vs_num;

    int bin_counter = 0;
    int axial_bins = 0 ;

    for (vs_num.segment_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_segment_num();
         ++vs_num.segment_num())
        axial_bins += this->proj_data_info_cyl_noarc_cor_sptr->get_num_axial_poss(vs_num.segment_num());

    const int total_bins =
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_views() * axial_bins *
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_tangential_poss();
    /* Currently, proj_data_info.find_cartesian_coordinates_of_detection() returns
     coordinate in a coordinate system where z=0 in the first ring of the scanner.
     We want to shift this to a coordinate system where z=0 in the middle
     of the scanner.
     We can use get_m() as that uses the 'middle of the scanner' system.
     (sorry)
  */
#ifndef NDEBUG
    {
        CartesianCoordinate3D<float> detector_coord_A, detector_coord_B;
        // check above statement
        this->proj_data_info_cyl_noarc_cor_sptr->find_cartesian_coordinates_of_detection(
                    detector_coord_A, detector_coord_B, Bin(0, 0, 0, 0));
        assert(detector_coord_A.z() == 0);
        assert(detector_coord_B.z() == 0);
        // check that get_m refers to the middle of the scanner
        const float m_first =
                this->proj_data_info_cyl_noarc_cor_sptr->get_m(Bin(0, 0, this->proj_data_info_cyl_noarc_cor_sptr->get_min_axial_pos_num(0), 0));
        const float m_last =
                this->proj_data_info_cyl_noarc_cor_sptr->get_m(Bin(0, 0, this->proj_data_info_cyl_noarc_cor_sptr->get_max_axial_pos_num(0), 0));
        assert(fabs(m_last + m_first) < m_last * 10E-4);
    }
#endif
    this->shift_detector_coordinates_to_origin =
            CartesianCoordinate3D<float>(this->proj_data_info_cyl_noarc_cor_sptr->get_m(Bin(0, 0, 0, 0)), 0, 0);
    float total_scatter = 0 ;

    info("ScatterSimulator: Initialization finished ...");

    for (vs_num.segment_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_segment_num();
         ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_view_num();
             vs_num.view_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_view_num();
             ++vs_num.view_num())
        {

            total_scatter += this->process_data_for_view_segment_num(vs_num);
            bin_counter +=
                    this->proj_data_info_cyl_noarc_cor_sptr->get_num_axial_poss(vs_num.segment_num()) *
                    this->proj_data_info_cyl_noarc_cor_sptr->get_num_tangential_poss();
           // info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);

            std::cout<< bin_counter << " / "<< total_bins <<std::endl;

        }
    }

    if (detection_points_vector.size() != static_cast<unsigned int>(total_detectors))
    {
        warning("Expected num detectors: %d, but found %d\n",
                total_detectors, detection_points_vector.size());
        return Succeeded::no;
    }


    std::cerr << "TOTAL SCATTER:= " << total_scatter << '\n';
    return Succeeded::yes;
}

//xxx double
double
ScatterSimulation::
process_data_for_view_segment_num(const ViewSegmentNumbers& vs_num)
{
    // First construct a vector of all bins that we'll process.
    // The reason for making this list before the actual calculation is that we can then parallelise over all bins
    // without having to think about double loops.
    std::vector<Bin> all_bins;
    {
        Bin bin(vs_num.segment_num(), vs_num.view_num(), 0, 0);

        for (bin.axial_pos_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_axial_pos_num(bin.segment_num());
             bin.axial_pos_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_axial_pos_num(bin.segment_num());
             ++bin.axial_pos_num())
        {
            for (bin.tangential_pos_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_tangential_pos_num();
                 bin.tangential_pos_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_tangential_pos_num();
                 ++bin.tangential_pos_num())
            {
                all_bins.push_back(bin);
            }
        }
    }
    // now compute scatter for all bins
    double total_scatter = 0;
    Viewgram<float> viewgram =
            this->output_proj_data_sptr->get_empty_viewgram(vs_num.view_num(), vs_num.segment_num());
#ifdef STIR_OPENMP
#pragma omp parallel for reduction(+:total_scatter) schedule(dynamic)
#endif

    for (int i = 0; i < static_cast<int>(all_bins.size()); ++i)
    {
        const Bin bin = all_bins[i];
        unsigned det_num_A = 0; // initialise to avoid compiler warnings
        unsigned det_num_B = 0;
        this->find_detectors(det_num_A, det_num_B, bin);
        const double scatter_ratio =
                scatter_estimate(det_num_A, det_num_B);
        viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] =
                static_cast<float>(scatter_ratio);
        total_scatter += scatter_ratio;


    } // end loop over bins

    if (this->output_proj_data_sptr->set_viewgram(viewgram) == Succeeded::no)
        error("ScatterEstimationByBin: error writing viewgram");

    return static_cast<double>(viewgram.sum());
}

void
ScatterSimulation::set_defaults()
{
    this->attenuation_threshold =  0.01f ;
    this->random = true;
    this->use_cache = true;
    this->zoom_xy = 1.f;
    this->zoom_z = 1.f;
    this->size_xy = -1;
    this->size_z = -1;
    this->downsample_scanner_dets = 0;
    this->downsample_scanner_rings = 0;
    this->density_image_filename = "";
    this->activity_image_filename = "";
    this->density_image_for_scatter_points_output_filename ="";
    this->density_image_for_scatter_points_filename = "";
    this->template_proj_data_filename = "";
    this->remove_cache_for_integrals_over_activity();
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
ask_parameters()
{
    this->attenuation_threshold = ask_num("attenuation threshold(cm^-1)",0.0f, 5.0f, 0.01f);
    this->random = ask_num("random points?",0, 0, 0);
    this->use_cache =  ask_num(" Use cache?",0, 1, 1);
    this->density_image_filename = ask_string("density image filename", "");
    this->activity_image_filename = ask_string("activity image filename", "");
    this->density_image_for_scatter_points_filename = ask_string("density image for scatter points filename", "");
    // if empty ... ask zoom_xy and zoom_z
    this->template_proj_data_filename = ask_string("Scanner ProjData filename", "");
}

void
ScatterSimulation::initialise_keymap()
{

    this->parser.add_start_key("Scatter Simulation Parameters");
    this->parser.add_stop_key("end Scatter Simulation Parameters");
    this->parser.add_key("template projdata filename",
                         &this->template_proj_data_filename);
    this->parser.add_key("attenuation image filename",
                         &this->density_image_filename);
    this->parser.add_key("attenuation image for scatter points filename",
                         &this->density_image_for_scatter_points_filename);
    this->parser.add_key("zoom XY for attenuation image for scatter points",
                         &this->zoom_xy);
    this->parser.add_key("zoom Z for attenuation image for scatter points",
                         &this->zoom_z);
    this->parser.add_key("size XY for attenuation image for scatter points",
                         &this->size_xy);
    this->parser.add_key("size Z for attenuation image for scatter points",
                         &this->size_z);
    this->parser.add_key("attenuation image for scatter points output filename",
                         &this->density_image_for_scatter_points_output_filename);
    this->parser.add_key("reduce number of detectors per ring by",
                         &this->downsample_scanner_dets);
    this->parser.add_key("reduce number of rings by",
                         &this->downsample_scanner_rings);
    this->parser.add_key("activity image filename",
                         &this->activity_image_filename);
    this->parser.add_key("attenuation threshold",
                         &this->attenuation_threshold);
    this->parser.add_key("output filename prefix",
                         &this->output_proj_data_filename);
    this->parser.add_key("random", &this->random);
    this->parser.add_key("use cache", &this->use_cache);
}


bool
ScatterSimulation::
post_processing()
{

    if (this->template_proj_data_filename.size() > 0)
        this->set_template_proj_data_info(this->template_proj_data_filename);

    if (this->activity_image_filename.size() > 0)
        this->set_activity_image(this->activity_image_filename);

    if (this->density_image_filename.size() > 0)
        this->set_density_image(this->density_image_filename);

    if ((zoom_xy!=1 || zoom_z != 1) &&
            this->density_image_filename.size()>0)
    {

         this->set_density_image_for_scatter_points_sptr(downsample_image(this->density_image_sptr));

        if(this->density_image_for_scatter_points_output_filename.size()>0)
            OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
                    write_to_file(density_image_for_scatter_points_output_filename,
                                  *this->density_image_for_scatter_points_sptr);
    }
    else if(this->density_image_for_scatter_points_filename.size() > 0)
        this->set_density_image_for_scatter_points(this->density_image_for_scatter_points_filename);

    if (this->output_proj_data_filename.size() > 0)
        this->set_output_proj_data(this->output_proj_data_filename);

    return false;
}

Succeeded
ScatterSimulation::
set_up()
{
//    if (!is_null_ptr())
//    this->set_template_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone());
//    //this->set_exam_info_sptr(template_proj_data_sptr->get_exam_info_ptr()->create_shared_clone());

    return Succeeded::yes;
}

void
ScatterSimulation::
set_activity_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >& arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the activity image");

    this->activity_image_sptr = arg;
    this->remove_cache_for_integrals_over_activity();
}

void
ScatterSimulation::
set_activity_image(const std::string& filename)
{
    this->activity_image_filename = filename;
    this->activity_image_sptr=
            read_from_file<DiscretisedDensity<3,float> >(filename);

    if (is_null_ptr(this->activity_image_sptr))
    {
        error(boost::format("ScatterSimulation: Error reading activity image %s") %
              this->activity_image_filename);
    }
    this->remove_cache_for_integrals_over_activity();
}

void
ScatterSimulation::
set_density_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >& arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the density image");
    this->density_image_sptr=arg;
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
set_density_image(const std::string& filename)
{
    this->density_image_filename=filename;
    this->density_image_sptr=
            read_from_file<DiscretisedDensity<3,float> >(filename);
    if (is_null_ptr(this->density_image_sptr))
    {
        error(boost::format("Error reading density image %s") %
              this->density_image_filename);
    }
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
set_density_image_for_scatter_points_sptr(const shared_ptr<DiscretisedDensity<3,float> >& arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the density image for scatter points.");
    this->density_image_for_scatter_points_sptr = arg;
    this->sample_scatter_points();
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
set_density_image_for_scatter_points(const std::string& filename)
{
    this->density_image_for_scatter_points_filename=filename;
    this->density_image_for_scatter_points_sptr=
            read_from_file<DiscretisedDensity<3,float> >(filename);

    if (is_null_ptr(this->density_image_for_scatter_points_sptr))
    {
        error(boost::format("Error reading density_for_scatter_points image %s") %
              this->density_image_for_scatter_points_filename);
    }
    this->sample_scatter_points();
    this->remove_cache_for_integrals_over_attenuation();
}

shared_ptr<DiscretisedDensity<3, float> >
ScatterSimulation::
downsample_image(shared_ptr<DiscretisedDensity<3, float> > arg,
                bool scale)
{
    int new_xy, new_z;
    VoxelsOnCartesianGrid<float>* tmp_image_ptr =
            dynamic_cast<VoxelsOnCartesianGrid<float>* >(arg->clone());

    new_xy = this->size_xy < 0 ? static_cast<int>(tmp_image_ptr->get_x_size() * zoom_xy ):
                                 size_xy;

    new_z = this->size_z < 0 ?  static_cast<int>(tmp_image_ptr->get_z_size() * zoom_z):
                                size_z;

    shared_ptr<VoxelsOnCartesianGrid<float> > tmp_image_lowres_sptr(new VoxelsOnCartesianGrid<float>());

    *tmp_image_lowres_sptr =
            zoom_image(*tmp_image_ptr,
                       CartesianCoordinate3D<float>(zoom_z, zoom_xy, zoom_xy),
                       CartesianCoordinate3D<float>(0.0f, 0.0f, 0.0f),
                       CartesianCoordinate3D<int>(new_z, new_xy, new_xy));

    // Scale values.
    if(scale)
    {
        float scale_value = this->zoom_xy * this->zoom_xy * this->zoom_z;
        *tmp_image_lowres_sptr *= scale_value;
    }

    return tmp_image_lowres_sptr;
}

void
ScatterSimulation::
set_output_proj_data_sptr(const shared_ptr<ExamInfo>& _exam,
                          const shared_ptr<ProjDataInfo>& _info,
                          const std::string & filename)
{
    if (filename.size() > 0 )
        this->output_proj_data_sptr.reset(new ProjDataInterfile(_exam,
                                                                _info,
                                                                filename));
    else
        this->output_proj_data_sptr.reset( new ProjDataInMemory(_exam,
                                                                _info));
}

shared_ptr<ProjData>
ScatterSimulation::
get_output_proj_data_sptr()
{

    if(is_null_ptr(this->output_proj_data_sptr))
    {
        this->output_proj_data_sptr.reset(new ProjDataInMemory(this->template_exam_info_sptr,
                                                                this->proj_data_info_cyl_noarc_cor_sptr->create_shared_clone()));
    }

    return this->output_proj_data_sptr;
}

void
ScatterSimulation::
set_output_proj_data(const std::string& filename)
{
    
    if(is_null_ptr(this->proj_data_info_cyl_noarc_cor_sptr))
    {
        error("Template ProjData has not been set. Abord.");
    }
    this->output_proj_data_filename = filename;
    if (is_null_ptr(this->template_exam_info_sptr))
    {
        shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
        this->output_proj_data_sptr.reset(new ProjDataInterfile(exam_info_sptr,
                                                                this->proj_data_info_cyl_noarc_cor_sptr->create_shared_clone(),
                                                                this->output_proj_data_filename,std::ios::in | std::ios::out | std::ios::trunc));
    }
    else
        this->output_proj_data_sptr.reset(new ProjDataInterfile(this->template_exam_info_sptr,
                                                                this->proj_data_info_cyl_noarc_cor_sptr->create_shared_clone(),
                                                                this->output_proj_data_filename,std::ios::in | std::ios::out | std::ios::trunc));
}

void
ScatterSimulation::
set_output_proj_data_sptr(shared_ptr<ProjData>& arg)
{
    this->output_proj_data_sptr = arg;
}

void
ScatterSimulation::
set_template_proj_data_info_sptr(const shared_ptr<ProjDataInfo>& arg)
{
    this->proj_data_info_cyl_noarc_cor_sptr.reset(dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(arg->clone()));

    if (is_null_ptr(this->proj_data_info_cyl_noarc_cor_sptr))
        error("ScatterSimulation: Can only handle non-arccorrected data");

    if (downsample_scanner_dets > 1 || downsample_scanner_rings > 1)
        this->proj_data_info_cyl_noarc_cor_sptr.reset(
                dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(downsample_scanner(proj_data_info_cyl_noarc_cor_sptr)->clone()) );


    if (this->proj_data_info_cyl_noarc_cor_sptr->get_num_segments() > 1) // do SSRB
    {
        info("ScatterSimulation: Performing SSRB on template info ...");
        this->proj_data_info_cyl_noarc_cor_sptr.reset(
                dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(SSRB(* this->proj_data_info_cyl_noarc_cor_sptr,
                                                                      this->proj_data_info_cyl_noarc_cor_sptr->get_num_segments(), 1, false)));
    }

    // find final size of detection_points_vector
    this->total_detectors =
            this->proj_data_info_cyl_noarc_cor_sptr->get_scanner_ptr()->get_num_rings()*
            this->proj_data_info_cyl_noarc_cor_sptr->get_scanner_ptr()->get_num_detectors_per_ring ();

    // reserve space to avoid reallocation, but the actual size will grow dynamically
    this->detection_points_vector.reserve(static_cast<std::size_t>(this->total_detectors));

    // remove any cached values as they'd be incorrect if the sizes changes
    this->remove_cache_for_integrals_over_attenuation();
    this->remove_cache_for_integrals_over_activity();
}

void
ScatterSimulation::
set_template_proj_data_info(const std::string& filename)
{
    this->template_proj_data_filename = filename;
    shared_ptr<ProjData> template_proj_data_sptr =
            ProjData::read_from_file(this->template_proj_data_filename);

    this->set_exam_info_sptr(template_proj_data_sptr->get_exam_info_sptr());

    shared_ptr<ProjDataInfo> tmp_proj_data_info_sptr =
            template_proj_data_sptr->get_proj_data_info_sptr();

    this->set_template_proj_data_info_sptr(tmp_proj_data_info_sptr);

}

void
ScatterSimulation::
set_exam_info_sptr(const shared_ptr<ExamInfo>& arg)
{
    this->template_exam_info_sptr = arg;
}

shared_ptr<ProjDataInfo>
ScatterSimulation::
downsample_scanner(shared_ptr<ProjDataInfo> arg)
{

    // copy localy.
    info("ScatterSimulator: Downsampling scanner of template info ...");
    shared_ptr<Scanner> new_scanner_sptr( new Scanner(*arg->get_scanner_ptr()));

    // preserve the lenght of the scanner
    float scanner_lenght = new_scanner_sptr->get_num_rings()* new_scanner_sptr->get_ring_spacing();

    new_scanner_sptr->set_num_rings(static_cast<int>(new_scanner_sptr->get_num_rings()/downsample_scanner_rings));
    new_scanner_sptr->set_num_detectors_per_ring(static_cast<int>(new_scanner_sptr->get_num_detectors_per_ring()/downsample_scanner_dets));
    new_scanner_sptr->set_ring_spacing(static_cast<float>(scanner_lenght/new_scanner_sptr->get_num_rings()));


    ProjDataInfo * tmp_proj_data_info_2d_ptr = ProjDataInfo::ProjDataInfoCTI(new_scanner_sptr,
                                                                             1, 0,
                                                                             new_scanner_sptr->get_num_detectors_per_ring()/2,
                                                                             new_scanner_sptr->get_num_detectors_per_ring()/2,
                                                                             false);

    return tmp_proj_data_info_2d_ptr->create_shared_clone();
}

void
ScatterSimulation::
set_attenuation_threshold(const float arg)
{
    attenuation_threshold = arg;
}

void
ScatterSimulation::
set_random_point(const bool arg)
{
    random = arg;
}

void
ScatterSimulation::
set_cache_enabled(const bool arg)
{
    use_cache = arg;
}




END_NAMESPACE_STIR
