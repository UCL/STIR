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
#include "stir/scatter/SingleScatterLikelihoodAndGradient.h"
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
#include "stir/NumericInfo.h"

START_NAMESPACE_STIR

const char * const
SingleScatterLikelihoodAndGradient::registered_name =
        "Single Scatter Likelihood And Gradient";


SingleScatterLikelihoodAndGradient::
SingleScatterLikelihoodAndGradient() :
    base_type()
{
    this->set_defaults();
}



SingleScatterLikelihoodAndGradient::
SingleScatterLikelihoodAndGradient(const std::string& parameter_filename)
{
    this->initialise(parameter_filename);
}

SingleScatterLikelihoodAndGradient::
~SingleScatterLikelihoodAndGradient()
{}

static const float total_Compton_cross_section_511keV =
ScatterSimulation::
total_Compton_cross_section(511.F);

double
SingleScatterLikelihoodAndGradient::
L_G_function(ProjData& data,VoxelsOnCartesianGrid<float>& gradient_image, const float rescale, bool isgradient)
{

if (isgradient)
     std::cerr <<"IS GRADIENT"<< "\n";

else  std::cerr <<"IS JACOBIAN"<< "\n";

    this->output_proj_data_sptr->fill(0.f);


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
    double sum = 0;

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

    info("ScatterSimulator: Initialization finished ...");

    for (vs_num.segment_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_segment_num();
         ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_view_num();
             vs_num.view_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_view_num();
             ++vs_num.view_num())
        {
            //info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);


            sum+=this->L_G_for_view_segment_number(data, gradient_image,vs_num,rescale,isgradient);

            bin_counter +=
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_axial_poss(vs_num.segment_num()) *
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_tangential_poss();
            //info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);

            std::cout<< bin_counter << " / "<< total_bins <<std::endl;

        }
    }

    int max_x = gradient_image.get_max_x();
    int min_x = gradient_image.get_min_x();

    std::cerr << "LIKELIHOOD:= " << sum << '\n';

    std::cerr << "MIN_GRADIENT_X:= " << min_x << '\n';
    std::cerr << "MAX_GRADIENT_X:= " << max_x << '\n';

    return sum;
}

double
SingleScatterLikelihoodAndGradient::
L_G_for_view_segment_number(ProjData&data,VoxelsOnCartesianGrid<float>& gradient_image,const ViewSegmentNumbers& vs_num, const float rescale, bool isgradient)
{

    Viewgram<float> viewgram=data.get_viewgram(vs_num.view_num(), vs_num.segment_num(),false);
    Viewgram<float> v_est= this->proj_data_info_cyl_noarc_cor_sptr->get_empty_viewgram(vs_num.view_num(), vs_num.segment_num());

    double sum = L_G_for_viewgram(viewgram,v_est,gradient_image, rescale, isgradient);

    return sum;

}
double
SingleScatterLikelihoodAndGradient::
L_G_for_viewgram(Viewgram<float>& viewgram,Viewgram<float>& v_est,VoxelsOnCartesianGrid<float>& gradient_image,const float rescale, bool isgradient)
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

       double sum = 0;


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
           const double y = L_G_estimate(tmp_gradient_image,bin,isgradient);

           v_est[bin.axial_pos_num()][bin.tangential_pos_num()] = static_cast<float>(rescale*y);


           sum+=viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]*log(v_est[bin.axial_pos_num()][bin.tangential_pos_num()]+0.000000000000000000001)- v_est[bin.axial_pos_num()][bin.tangential_pos_num()]-0.000000000000000000001;


           if (isgradient)
           {

               gradient_image += tmp_gradient_image*(viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]/(v_est[bin.axial_pos_num()][bin.tangential_pos_num()]+0.000000000000000000001)-1);
            }
           else
               {

               gradient_image += tmp_gradient_image;
                }
       }

       return sum;
}


double
SingleScatterLikelihoodAndGradient::
L_G_function_from_est_data(ProjData& data,ProjData& est_data,VoxelsOnCartesianGrid<float>& gradient_image, const float rescale)
{


    this->output_proj_data_sptr->fill(0.f);


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
    double sum = 0;

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

    info("ScatterSimulator: Initialization finished ...");

    for (vs_num.segment_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_segment_num();
         ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_view_num();
             vs_num.view_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_view_num();
             ++vs_num.view_num())
        {
            //info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);


            sum+=this->L_G_for_view_segment_number_from_est_data(data, est_data,gradient_image,vs_num,rescale);

            bin_counter +=
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_axial_poss(vs_num.segment_num()) *
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_tangential_poss();
            //info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);

            std::cout<< bin_counter << " / "<< total_bins <<std::endl;

        }
    }

    int max_x = gradient_image.get_max_x();
    int min_x = gradient_image.get_min_x();

    std::cerr << "LIKELIHOOD:= " << sum << '\n';

    std::cerr << "MIN_GRADIENT_X:= " << min_x << '\n';
    std::cerr << "MAX_GRADIENT_X:= " << max_x << '\n';

    return sum;
}

double
SingleScatterLikelihoodAndGradient::
L_G_for_view_segment_number_from_est_data(ProjData&data,ProjData&est_data,VoxelsOnCartesianGrid<float>& gradient_image,const ViewSegmentNumbers& vs_num, const float rescale)
{

    Viewgram<float> viewgram=data.get_viewgram(vs_num.view_num(), vs_num.segment_num(),false);
    Viewgram<float> v_est= est_data.get_viewgram(vs_num.view_num(), vs_num.segment_num(),false);

    double sum = L_G_for_viewgram_from_est_data(viewgram,v_est,gradient_image, rescale);

    return sum;

}
double
SingleScatterLikelihoodAndGradient::
L_G_for_viewgram_from_est_data(Viewgram<float>& viewgram,Viewgram<float>& v_est,VoxelsOnCartesianGrid<float>& gradient_image,const float rescale)
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

       double sum = 0;


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
           const double y = L_G_estimate(tmp_gradient_image,bin,true);



           sum+=viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]*log(v_est[bin.axial_pos_num()][bin.tangential_pos_num()])- v_est[bin.axial_pos_num()][bin.tangential_pos_num()];




               gradient_image += tmp_gradient_image*(viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]/(v_est[bin.axial_pos_num()][bin.tangential_pos_num()])-1);

       }

       return sum;
}


double
SingleScatterLikelihoodAndGradient::
L_G_estimate(VoxelsOnCartesianGrid<float>& gradient_image_bin,const Bin bin, bool isgradient)
{
    double scatter_ratio_singles = 0;
    unsigned det_num_B=0;
    unsigned det_num_A=0;

    this->find_detectors(det_num_A, det_num_B,bin);

    for(std::size_t scatter_point_num =0;
        scatter_point_num < this->scatt_points_vector.size();
        ++scatter_point_num)
    {




        scatter_ratio_singles +=
        L_G_for_one_scatter_point(gradient_image_bin,
                                  scatter_point_num,
                                  det_num_A, det_num_B, isgradient);

    }

    return scatter_ratio_singles;
}



float
SingleScatterLikelihoodAndGradient::
L_G_for_one_scatter_point(VoxelsOnCartesianGrid<float>& gradient,
        const std::size_t scatter_point_num,
        const unsigned det_num_A,
        const unsigned det_num_B, bool isgradient)
{

    // The code now supports more than one energy window: the low energy threshold has to correspond to lowest window.

    int low = 0;

    if (this->template_exam_info_sptr->get_num_energy_windows()>1)

    {

        int first_window=this->template_exam_info_sptr->get_energy_window_pair().first-1;
        int second_window=this->template_exam_info_sptr->get_energy_window_pair().second-1;

        if(this->template_exam_info_sptr->get_low_energy_thres(first_window) <= this->template_exam_info_sptr->get_low_energy_thres(second_window) )

        {
            low = first_window;
        }

        else if(this->template_exam_info_sptr->get_low_energy_thres(first_window) >= this->template_exam_info_sptr->get_low_energy_thres(second_window) )

        {
             low = second_window;
        }

    }

    static const float max_single_scatter_cos_angle=max_cos_angle(this->template_exam_info_sptr->get_low_energy_thres(low),
            2.f,
            this->proj_data_info_cyl_noarc_cor_sptr->get_scanner_ptr()->get_energy_resolution());

    //static const float min_energy=energy_lower_limit(lower_energy_threshold,2.,energy_resolution);

    const CartesianCoordinate3D<float>& scatter_point =
    this->scatt_points_vector[scatter_point_num].coord;
    const CartesianCoordinate3D<float>& detector_coord_A =
    this->detection_points_vector[det_num_A];
    const CartesianCoordinate3D<float>& detector_coord_B =
    this->detection_points_vector[det_num_B];
    // note: costheta is -cos_angle such that it is 1 for zero scatter angle
    const float costheta = static_cast<float>(
            -cos_angle(detector_coord_A - scatter_point,
                    detector_coord_B - scatter_point));
    // note: costheta is identical for scatter to A or scatter to B
    // Hence, the Compton_cross_section and energy are identical for both cases as well.
    if(max_single_scatter_cos_angle>costheta)
    return 0;
    const float new_energy =
    photon_energy_after_Compton_scatter_511keV(costheta);

    // The detection efficiency varies with respect to the energy window.
    //The code can now compute the scatter for a combination of two windows X and Y
    //Default: one window -> The code will combine the window with itself

    std::vector<float>detection_efficiency_scattered;
    std::vector<float>detection_efficiency_unscattered;

    for (int i = 0; i < this->template_exam_info_sptr->get_num_energy_windows(); ++i)
    {
        detection_efficiency_scattered.push_back(0);
        detection_efficiency_unscattered.push_back(0);

    }

    for (int i = 0; i < this->template_exam_info_sptr->get_num_energy_windows(); ++i)
    {
        detection_efficiency_scattered[i] = detection_efficiency(new_energy,i);
        detection_efficiency_unscattered[i] = detection_efficiency(511.F,i);

        if (detection_efficiency_scattered[i]==0)
        return 0;
        if (detection_efficiency_unscattered[i]==0)
        return 0;
    }

    //compute the probability of detection for two given energy windows X and Y

    int index0 = 0;
    int index1 = 0;

    if (this->template_exam_info_sptr->get_num_energy_windows()>1)
    {

        index0 = this->template_exam_info_sptr->get_energy_window_pair().first-1;
        index1 = this->template_exam_info_sptr->get_energy_window_pair().second-1;

    }

    float detection_probability_XY=detection_efficiency_scattered[index0]*detection_efficiency_unscattered[index1];
    float detection_probability_YX=detection_efficiency_scattered[index1]*detection_efficiency_unscattered[index0];

    const float emiss_to_detA =
    cached_integral_over_activity_image_between_scattpoint_det
    (static_cast<unsigned int> (scatter_point_num),
            det_num_A);
    const float emiss_to_detB =
    cached_integral_over_activity_image_between_scattpoint_det
    (static_cast<unsigned int> (scatter_point_num),
            det_num_B);
    if (emiss_to_detA==0 && emiss_to_detB==0)
    return 0;
    const float atten_to_detA =
    cached_exp_integral_over_attenuation_image_between_scattpoint_det
    (scatter_point_num,
            det_num_A);
    const float atten_to_detB =
    cached_exp_integral_over_attenuation_image_between_scattpoint_det
    (scatter_point_num,
            det_num_B);

    const float dif_Compton_cross_section_value =
    dif_Compton_cross_section(costheta, 511.F);

    const float rA_squared=static_cast<float>(norm_squared(scatter_point-detector_coord_A));
    const float rB_squared=static_cast<float>(norm_squared(scatter_point-detector_coord_B));

    const float scatter_point_mu=
    scatt_points_vector[scatter_point_num].mu_value;

    const CartesianCoordinate3D<float>
    detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
    const CartesianCoordinate3D<float>
    detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
    const float cos_incident_angle_AS = static_cast<float>(
            cos_angle(scatter_point - detector_coord_A,
                    detA_to_ring_center));
    const float cos_incident_angle_BS = static_cast<float>(
            cos_angle(scatter_point - detector_coord_B,
                    detB_to_ring_center));

#ifndef NDEBUG
    {
        // check if mu-value ok
        // currently terribly shift needed as in sample_scatter_points (TODO)
        const VoxelsOnCartesianGrid<float>& image =
        dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*this->density_image_for_scatter_points_sptr);
        const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
        const float z_to_middle =
        (image.get_max_index() + image.get_min_index())*voxel_size.z()/2.F;
        CartesianCoordinate3D<float> shifted=scatter_point;
        shifted.z() += z_to_middle;
        assert(scatter_point_mu==
                (*this->density_image_for_scatter_points_sptr)[this->density_image_for_scatter_points_sptr->get_indices_closest_to_physical_coordinates(shifted)]);
    }
#endif

#ifndef NEWSCALE
    /* projectors work in pixel units, so convert attenuation data
     from cm^-1 to pixel_units^-1 */
    const float rescale =
    dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float> &>(*density_image_sptr).
    get_grid_spacing()[3]/10;
#else
    const float rescale =
    0.1F;
#endif


    //normalisation


      // we will divide by the effiency of the detector pair for unscattered photons
      // (computed with the same detection model as used in the scatter code)
      // This way, the scatter estimate will correspond to a 'normalised' scatter estimate.

      // there is a scatter_volume factor for every scatter point, as the sum over scatter points
      // is an approximation for the integral over the scatter point.

      // the factors total_Compton_cross_section_511keV should probably be moved to the scatter_computation code


     // currently the scatter simulation is normalised w.r.t. the detection efficiency in the photopeak window
      //find the window that contains 511 keV

      int index_photopeak = 0; //default for one energy window

      if (this->template_exam_info_sptr->get_num_energy_windows()>1)
      {
         for (int i = 0 ; i < this->template_exam_info_sptr->get_num_energy_windows() ; ++i)
         {
                if( this->template_exam_info_sptr->get_high_energy_thres(i) >= 511.F &&  this->template_exam_info_sptr->get_low_energy_thres(i) <= 511.F)

                {

                     index_photopeak = i;
                  }

         }
       }

      //normalisation factor between trues and scattered counts

     const double common_factor =
            1/detection_efficiency_no_scatter(det_num_A, det_num_B, index_photopeak) *
            scatter_volume/total_Compton_cross_section_511keV;


    // Single ScatterForward Model

    float scatter_ratio=0;

    scatter_ratio= (detection_probability_XY*emiss_to_detA*(1.F/rB_squared)*pow(atten_to_detB,total_Compton_cross_section_relative_to_511keV(new_energy)-1)
                    +detection_probability_YX*emiss_to_detB*(1.F/rA_squared)*pow(atten_to_detA,total_Compton_cross_section_relative_to_511keV(new_energy)-1))
                    *atten_to_detB
                    *atten_to_detA
                    *scatter_point_mu
                    *cos_incident_angle_AS
                    *cos_incident_angle_BS
                    *dif_Compton_cross_section_value
                    *common_factor;

    /*Single Scatter Forward model Jacobian:
     * The derivative is given by three term, respectively in [A,S], [B,S] and [S] */

    float contribution_AS = (detection_probability_XY*emiss_to_detA*(1.F/rB_squared)*pow(atten_to_detB,total_Compton_cross_section_relative_to_511keV(new_energy)-1)
                            +detection_probability_YX*emiss_to_detB*(1.F/rA_squared)*pow(atten_to_detA,total_Compton_cross_section_relative_to_511keV(new_energy)-1)*
                            total_Compton_cross_section_relative_to_511keV(new_energy))
                            *atten_to_detB
                            *atten_to_detA
                            *scatter_point_mu
                            *cos_incident_angle_AS
                            *cos_incident_angle_BS
                            *dif_Compton_cross_section_value
                            *common_factor;

    float contribution_BS = (detection_probability_XY*emiss_to_detA*
                            (1.F/rB_squared)*pow(atten_to_detB,total_Compton_cross_section_relative_to_511keV(new_energy)-1)*
                            (total_Compton_cross_section_relative_to_511keV(new_energy))
                            +detection_probability_YX*emiss_to_detB*(1.F/rA_squared)*
                             pow(atten_to_detA,total_Compton_cross_section_relative_to_511keV(new_energy)-1))
                            *atten_to_detB
                            *atten_to_detA
                            *scatter_point_mu
                            *cos_incident_angle_AS
                            *cos_incident_angle_BS
                            *dif_Compton_cross_section_value
                            *common_factor;

    float contribution_S = (detection_probability_XY*emiss_to_detA*(1.F/rB_squared)*pow(atten_to_detB,total_Compton_cross_section_relative_to_511keV(new_energy)-1)
                            +detection_probability_YX*emiss_to_detB*(1.F/rA_squared)*pow(atten_to_detA,total_Compton_cross_section_relative_to_511keV(new_energy)-1))
                            *atten_to_detB
                            *atten_to_detA
                            *cos_incident_angle_AS
                            *cos_incident_angle_BS
                            *dif_Compton_cross_section_value
                            *common_factor;

    //Fill gradient image along [A,S], [B,S] and in [S]


    line_contribution(gradient,rescale,scatter_point,
            detector_coord_B,contribution_BS);

    line_contribution(gradient,rescale,scatter_point,
            detector_coord_A,contribution_AS);

    s_contribution(gradient,scatter_point,
            contribution_S);


    return scatter_ratio;

}

END_NAMESPACE_STIR

