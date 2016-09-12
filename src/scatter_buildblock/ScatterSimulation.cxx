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

#include "stir/stir_math.h"
#include "stir/zoom.h"
#include "stir/ArrayFunction.h"
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

    this->output_proj_data_sptr->fill(0.f);

    // The activiy image has been changed, from another class.
    this->remove_cache_for_integrals_over_activity();

    this->initialise_cache_for_scattpoint_det_integrals_over_attenuation();
    this->initialise_cache_for_scattpoint_det_integrals_over_activity();
    ViewSegmentNumbers vs_num;

    int bin_counter = 0;
    int axial_bins = 0 ;

    for (vs_num.segment_num() = this->proj_data_info_ptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_ptr->get_max_segment_num();
         ++vs_num.segment_num())
        axial_bins += this->proj_data_info_ptr->get_num_axial_poss(vs_num.segment_num());

    const int total_bins =
            this->proj_data_info_ptr->get_num_views() * axial_bins *
            this->proj_data_info_ptr->get_num_tangential_poss();
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
        this->proj_data_info_ptr->find_cartesian_coordinates_of_detection(
                    detector_coord_A, detector_coord_B, Bin(0, 0, 0, 0));
        assert(detector_coord_A.z() == 0);
        assert(detector_coord_B.z() == 0);
        // check that get_m refers to the middle of the scanner
        const float m_first =
                this->proj_data_info_ptr->get_m(Bin(0, 0, this->proj_data_info_ptr->get_min_axial_pos_num(0), 0));
        const float m_last =
                this->proj_data_info_ptr->get_m(Bin(0, 0, this->proj_data_info_ptr->get_max_axial_pos_num(0), 0));
        assert(fabs(m_last + m_first) < m_last * 10E-4);
    }
#endif
    this->shift_detector_coordinates_to_origin =
            CartesianCoordinate3D<float>(this->proj_data_info_ptr->get_m(Bin(0, 0, 0, 0)), 0, 0);
    float total_scatter = 0 ;

    for (vs_num.segment_num() = this->proj_data_info_ptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_ptr->get_max_segment_num();
         ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_ptr->get_min_view_num();
             vs_num.view_num() <= this->proj_data_info_ptr->get_max_view_num();
             ++vs_num.view_num())
        {
            total_scatter += this->process_data_for_view_segment_num(vs_num);
            bin_counter +=
                    this->proj_data_info_ptr->get_num_axial_poss(vs_num.segment_num()) *
                    this->proj_data_info_ptr->get_num_tangential_poss();
        }
    }

    if (detection_points_vector.size() != static_cast<unsigned int>(total_detectors))
    {
        warning("Expected num detectors: %d, but found %d\n",
                total_detectors, detection_points_vector.size());
        return Succeeded::no;
    }

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

        for (bin.axial_pos_num() = this->proj_data_info_ptr->get_min_axial_pos_num(bin.segment_num());
             bin.axial_pos_num() <= this->proj_data_info_ptr->get_max_axial_pos_num(bin.segment_num());
             ++bin.axial_pos_num())
        {
            for (bin.tangential_pos_num() = this->proj_data_info_ptr->get_min_tangential_pos_num();
                 bin.tangential_pos_num() <= this->proj_data_info_ptr->get_max_tangential_pos_num();
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
    this->density_image_filename = "";
    this->density_image_for_scatter_points_filename = "";
    this->scatter_proj_data_filename = "";
    this->remove_cache_for_integrals_over_activity();
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
ask_parameters()
{

}

void
ScatterSimulation::initialise_keymap()
{
    this->parser.add_start_key("Scatter Simulation Parameters");
    this->parser.add_stop_key("end Scatter Simulation Parameters");
    this->parser.add_key("scatter projdata filename",
                         &this->scatter_proj_data_filename);
    this->parser.add_key("attenuation image filename",
                         &this->density_image_filename);
    this->parser.add_key("attenuation image for scatter points filename",
                         &this->density_image_for_scatter_points_filename);
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
    if (this->scatter_proj_data_filename.size() > 0)
        this->set_scatter_proj_data_info(this->scatter_proj_data_filename);

    if (this->activity_image_filename.size() > 0)
        this->set_activity_image(this->activity_image_filename);

    if (this->density_image_filename.size() > 0)
        this->set_density_image(this->density_image_filename);

    if(this->density_image_for_scatter_points_filename.size() > 0 && !this->recompute_sub_atten_image)
        this->set_density_image_for_scatter_points(this->density_image_for_scatter_points_filename);

    return false;
}

END_NAMESPACE_STIR
