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
#include "stir/scatter/SingleScatterLikelihoodAndGradient.h"
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



double
SingleScatterLikelihoodAndGradient::
L_G_function(ProjData& data,VoxelsOnCartesianGrid<float>& gradient_image, const float scale_factor, bool isgradient)
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
    
    
    for (int i = 0; i < this->template_exam_info_sptr->get_num_energy_windows(); ++i)
    {
        std::cerr << "energy window lower level"<<"["<<i+1<<"] := "<< this->template_exam_info_sptr->get_low_energy_thres(i) << '\n';
        std::cerr << "energy window upper level"<<"["<<i+1<<"] := "<<  this->template_exam_info_sptr->get_high_energy_thres(i) << '\n';
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
    float sum = 0 ;
    
    info("ScatterSimulator: Initialization finished ...");
    
    for (vs_num.segment_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_segment_num();
         ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_view_num();
             vs_num.view_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_view_num();
             ++vs_num.view_num())
        {
            info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);
            sum+=this->L_G_for_view_segment_number(data, gradient_image,vs_num,scale_factor,isgradient);
            bin_counter +=
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_axial_poss(vs_num.segment_num()) *
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_tangential_poss();
            info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);
            
            
            
        }
    }
    
    if (detection_points_vector.size() != static_cast<unsigned int>(total_detectors))
    {
        warning("Expected num detectors: %d, but found %d\n",
                total_detectors, detection_points_vector.size());
        return Succeeded::no;
    }
    
    
    std::cerr << "LIKELIHOOD:= " << sum << '\n';
    return Succeeded::yes;
}

double
SingleScatterLikelihoodAndGradient::
L_G_for_view_segment_number(ProjData&data,VoxelsOnCartesianGrid<float>& gradient_image,const ViewSegmentNumbers& vs_num, const float scale_factor, bool isgradient)
{
    Viewgram<float> viewgram=data.get_viewgram(vs_num.view_num(), vs_num.segment_num(),false);
    Viewgram<float> v_est= this->proj_data_info_cyl_noarc_cor_sptr->get_empty_viewgram(vs_num.view_num(), vs_num.segment_num());
    
    double sum = L_G_for_viewgram(viewgram,v_est,gradient_image, scale_factor, isgradient);
    
    return sum;
    
}
double
SingleScatterLikelihoodAndGradient::
L_G_for_viewgram(Viewgram<float>& viewgram,Viewgram<float>& v_est,VoxelsOnCartesianGrid<float>& gradient_image,const float scale_factor, bool isgradient)
{

	const ViewSegmentNumbers vs_num(viewgram.get_view_num(),viewgram.get_segment_num());

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

       double sum =0;


       VoxelsOnCartesianGrid<float> tmp_gradient_image(gradient_image);

#ifdef STIR_OPENMP
#pragma omp parallel for reduction(+:total_scatter) schedule(dynamic)
#endif
    
       for (int i = 0; i < static_cast<int>(all_bins.size()); ++i)
       {
    //creates a template image to fill
    	   tmp_gradient_image.fill(0);

           const Bin bin = all_bins[i];

    //forward model
           const double y = L_G_estimate(tmp_gradient_image,bin);

           v_est[bin.axial_pos_num()][bin.tangential_pos_num()] = static_cast<float>(scale_factor*y);


           sum+=viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]*log(v_est[bin.axial_pos_num()][bin.tangential_pos_num()]+0.000000000000000000001)- v_est[bin.axial_pos_num()][bin.tangential_pos_num()]-0.000000000000000000001;


           if (isgradient)
        	   gradient_image += tmp_gradient_image*(viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]/(v_est[bin.axial_pos_num()][bin.tangential_pos_num()]+0.000000000000000000001)-1);


       }

       return sum;
}


END_NAMESPACE_STIR
