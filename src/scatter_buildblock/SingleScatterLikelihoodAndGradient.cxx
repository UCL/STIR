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
#include "stir/scatter/ScatterEstimation.h"
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
L_G_function(const ProjData& data,VoxelsOnCartesianGrid<float>& gradient_image, const bool compute_gradient, const bool isgradient_mu, const float rescale)
{

    shared_ptr<ProjData> add_sino(new ProjDataInMemory(this->output_proj_data_sptr->get_exam_info_sptr(),
                                                            this->output_proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone()));
    add_sino->fill(0.000000000000000000001); //automatically fills an additive sinogram to avoid division by 0
    L_G_function(data,*add_sino,gradient_image,compute_gradient,isgradient_mu,rescale);
}

double
SingleScatterLikelihoodAndGradient::
L_G_function(const ProjData& data,const ProjData &add_sino,VoxelsOnCartesianGrid<float>& gradient_image,const bool compute_gradient, const bool isgradient_mu, const float rescale)
{

    this->output_proj_data_sptr->fill(0.f);

    std::cout << "number of energy windows:= "<<  this->template_exam_info_sptr->get_num_energy_windows() << '\n';

    if(this->template_exam_info_sptr->get_energy_window_pair().first!= -1 &&
       this->template_exam_info_sptr->get_energy_window_pair().second!= -1 )
    {
        std::cout << "energy window pair :="<<" {"<<  this->template_exam_info_sptr->get_energy_window_pair().first  << ',' <<  this->template_exam_info_sptr->get_energy_window_pair().second <<"}\n";
    }


    for (int i = 0; i < this->template_exam_info_sptr->get_num_energy_windows(); ++i)
    {
        std::cout << "energy window lower level"<<"["<<i+1<<"] := "<< this->template_exam_info_sptr->get_low_energy_thres(i) << '\n';
        std::cout << "energy window upper level"<<"["<<i+1<<"] := "<<  this->template_exam_info_sptr->get_high_energy_thres(i) << '\n';
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


            sum+=this->L_G_for_view_segment_number(data, add_sino,gradient_image,vs_num,rescale,compute_gradient,isgradient_mu);

            bin_counter +=
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_axial_poss(vs_num.segment_num()) *
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_tangential_poss();
            //info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);

            std::cout<< bin_counter << " / "<< total_bins <<std::endl;

        }
    }
    std::cout << "LIKELIHOOD:= " << sum << '\n';
    return sum;
}

double
SingleScatterLikelihoodAndGradient::
L_G_for_view_segment_number(const ProjData&data, const ProjData&add_sino,VoxelsOnCartesianGrid<float>& gradient_image,const ViewSegmentNumbers& vs_num, const float rescale, const bool compute_gradient,const bool isgradient_mu)
{

    Viewgram<float> viewgram=data.get_viewgram(vs_num.view_num(), vs_num.segment_num(),false);
    Viewgram<float> v_add=add_sino.get_viewgram(vs_num.view_num(), vs_num.segment_num(),false);
    Viewgram<float> v_est= this->proj_data_info_cyl_noarc_cor_sptr->get_empty_viewgram(vs_num.view_num(), vs_num.segment_num());

    double sum = L_G_for_viewgram(viewgram,v_add,v_est,gradient_image, rescale, compute_gradient,isgradient_mu);
    return sum;

}

inline float KL(const double a, const float b, const float threshold_a = 0)
{
    assert(a>=0);
     assert(b>=0);
     float res = a<=threshold_a ? b : (a*(log(a)-log(b)) + b - a);
#ifndef NDEBUG
#define ICHANGEDIT
#define NDEBUG
     if (res != res)
       warning("KL nan at a=%g b=%g, threshold %g\n",a,b,threshold_a);
     if (res > 1.E20)
       warning("KL large at a=%g b=%g, threshold %g\n",a,b,threshold_a);
#ifdef ICHANGEDIT
#undef NDEBUG
#endif
#endif
     assert(res>=-1.e-4);
     return res;
}


double
SingleScatterLikelihoodAndGradient::
L_G_for_viewgram(const Viewgram<float>& viewgram, const Viewgram<float>& v_add,Viewgram<float>& v_est,VoxelsOnCartesianGrid<float>& gradient_image,const float rescale,const bool compute_gradient, const bool isgradient_mu)
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

       for (int i = 0; i < static_cast<int>(all_bins.size()); ++i)
       {
    //creates a template image to fill
           tmp_gradient_image.fill(0);

           const Bin bin = all_bins[i];

    //forward model
           const double y = L_G_estimate(tmp_gradient_image,bin,compute_gradient,isgradient_mu);

           v_est[bin.axial_pos_num()][bin.tangential_pos_num()] = static_cast<float>(rescale*y);
           //in case a scaling factor for the data is needed,i.e. for adding different level of noise. By default is set to 1.

           float eps = v_add[bin.axial_pos_num()][bin.tangential_pos_num()];
           sum+=viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]*log(v_est[bin.axial_pos_num()][bin.tangential_pos_num()]+eps)- v_est[bin.axial_pos_num()][bin.tangential_pos_num()]-eps;
           gradient_image += tmp_gradient_image*(viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]/(v_est[bin.axial_pos_num()][bin.tangential_pos_num()]+eps)-1);
       }

       return sum;
}




double
SingleScatterLikelihoodAndGradient::
L_G_estimate(VoxelsOnCartesianGrid<float>& gradient_image_bin,const Bin bin, const bool compute_gradient,const bool isgradient_mu)
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
                                  det_num_A, det_num_B,compute_gradient, isgradient_mu);
    }

    return scatter_ratio_singles;
}



float
SingleScatterLikelihoodAndGradient::
L_G_for_one_scatter_point(VoxelsOnCartesianGrid<float>& gradient,
        const std::size_t scatter_point_num,
        const unsigned det_num_A,
        const unsigned det_num_B, const bool compute_gradient,const bool isgradient_mu)
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


    //compute the probability of detection for two given energy windows X and Y


    std::vector<float>detection_efficiency_scattered;
    std::vector<float>detection_efficiency_unscattered;


    detection_efficiency_scattered.push_back(0);
    detection_efficiency_unscattered.push_back(0);



    //detection efficiency of each window for that energy
      for (int i = 0; i < this->template_exam_info_sptr->get_num_energy_windows(); ++i)
      {
          detection_efficiency_scattered[i] = detection_efficiency(new_energy,i);
          detection_efficiency_unscattered[i] = detection_efficiency(511.F,i);
      }


        int index0 = 0;
        int index1 = 0;

        if (this->template_exam_info_sptr->get_num_energy_windows()>1)
        {
            index0 = this->template_exam_info_sptr->get_energy_window_pair().first-1;
            index1 = this->template_exam_info_sptr->get_energy_window_pair().second-1;

        }


        float detection_probability_XY=detection_efficiency_scattered[index0]*detection_efficiency_unscattered[index1];
        float detection_probability_YX=detection_efficiency_scattered[index1]*detection_efficiency_unscattered[index0];


    if ((detection_probability_XY==0)&&(detection_probability_YX==0))
      return 0;

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

    if (cos_incident_angle_AS*cos_incident_angle_BS<0)
    return 0;

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
      // we will divide by the solid angle factors for unscattered photons
      // (computed with the same detection model as used in the scatter code)
      // the energy dependency is left out


    const double common_factor =
        1/detection_efficiency_no_scatter(det_num_A, det_num_B) *
        scatter_volume/total_Compton_cross_section_511keV;


    // Single ScatterForward Model

    const double line_integral1 = detection_probability_XY*(1.F/rB_squared)*pow(atten_to_detB,total_Compton_cross_section_relative_to_511keV(new_energy)-1);
    const double line_integral2 = detection_probability_YX*(1.F/rA_squared)*pow(atten_to_detA,total_Compton_cross_section_relative_to_511keV(new_energy)-1);
    const double line_integral1_times_activityA = line_integral1*emiss_to_detA;
    const double line_integral2_times_activityB = line_integral2*emiss_to_detB;
    const double global_factor = atten_to_detB*atten_to_detA*cos_incident_angle_AS*cos_incident_angle_BS*dif_Compton_cross_section_value*common_factor;
    const double global_factor_times_mu = global_factor*scatter_point_mu;

    float scatter_ratio=0;

    scatter_ratio = (line_integral1_times_activityA+line_integral2_times_activityB)*global_factor_times_mu;


    /*Single Scatter Forward model Jacobian w.r.t. attenuation:
     * The derivative is given by three terms, respectively in [A,S], [B,S] and [S] */

    float contribution_AS_mu = (line_integral1_times_activityA+line_integral2_times_activityB*total_Compton_cross_section_relative_to_511keV(new_energy))
                               *global_factor_times_mu;

    float contribution_BS_mu = (line_integral1_times_activityA*total_Compton_cross_section_relative_to_511keV(new_energy)+line_integral2_times_activityB*pow(atten_to_detA,total_Compton_cross_section_relative_to_511keV(new_energy)-1))
                               *global_factor_times_mu;

    float contribution_S =      (line_integral1_times_activityA+line_integral2_times_activityB)
                                *global_factor;


    /*Single Scatter Forward model Jacobian w.r.t. activity:
     * The derivative is given by two terms, respectively in [A,S] and [B,S]  */


    float contribution_AS_act = line_integral1*global_factor_times_mu;
    float contribution_BS_act = line_integral2*global_factor_times_mu;

    //Fill gradient image along [A,S], [B,S] and in [S]

if(compute_gradient)
{
    if (isgradient_mu)

        {
            line_contribution(gradient,rescale,scatter_point,detector_coord_B,contribution_BS_mu);
            line_contribution(gradient,rescale,scatter_point, detector_coord_A,contribution_AS_mu);
            s_contribution(gradient,scatter_point,contribution_S);
        }
    else

        {
            line_contribution_act(gradient,scatter_point,detector_coord_B,contribution_BS_act);
            line_contribution_act(gradient,scatter_point, detector_coord_A,contribution_AS_act);
        }

}
    return scatter_ratio;
}

ProjDataInMemory
SingleScatterLikelihoodAndGradient::
likelihood_and_gradient_scatter(const ProjData &projdata, const ProjData& norm , const ProjData &add_projdata, VoxelsOnCartesianGrid<float>& gradient_image_HR, VoxelsOnCartesianGrid<float>& gradient_image_LR,const bool compute_gradient, const bool isgradient_mu)
{
    gradient_image_LR.fill(0);
    gradient_image_HR.fill(0);
    int length = this->output_proj_data_sptr->get_num_views()*this->output_proj_data_sptr->get_num_axial_poss(0)*this->output_proj_data_sptr->get_num_tangential_poss(); //TODO: the code is for segment zero only
    std::vector<VoxelsOnCartesianGrid<float> > jacobian_array;
    std::vector<float> ratio;
    jacobian_array.reserve(length);
    ratio.reserve(length);
    for (int i = 0 ; i < length ; ++i)
    {
       jacobian_array.push_back(gradient_image_LR);
       ratio.push_back(0);
    }

    const ProjDataInMemory est_data_LR = get_jacobian(jacobian_array, compute_gradient, isgradient_mu);
    const ProjDataInMemory est_data_HR = get_ratio(projdata,norm,add_projdata,est_data_LR,ratio);

    for (int i = 0 ; i < length ; ++i)
    {
    gradient_image_LR += jacobian_array[i]*ratio[i];
    }

    if((gradient_image_HR.get_x_size()!=gradient_image_LR.get_x_size())||(gradient_image_HR.get_y_size()!=gradient_image_LR.get_y_size())||(gradient_image_HR.get_z_size()!=gradient_image_LR.get_z_size()))
    {
        if(isgradient_mu==true)
        transpose_zoom_image(gradient_image_HR,gradient_image_LR,ZoomOptions::preserve_values);
        else
        transpose_zoom_image(gradient_image_HR,gradient_image_LR,ZoomOptions::preserve_projections);

    }

    return est_data_HR;

}

ProjDataInMemory
SingleScatterLikelihoodAndGradient::
get_jacobian(std::vector<VoxelsOnCartesianGrid<float> > &gradient_image_array,const bool compute_gradient, const bool isgradient_mu)
{

    ProjDataInMemory est_data(this->get_output_proj_data_sptr()->get_exam_info_sptr(),this->get_output_proj_data_sptr()->get_proj_data_info_sptr());
    est_data.fill(0);

    this->remove_cache_for_integrals_over_activity();
    this->remove_cache_for_integrals_over_attenuation();
    this->sample_scatter_points();
    this->initialise_cache_for_scattpoint_det_integrals_over_attenuation();
    this->initialise_cache_for_scattpoint_det_integrals_over_activity();

    int bin_counter = 0;
    int axial_bins = 0 ;
    double sum = 0;

//    #ifdef STIR_OPENMP
//    #pragma omp parallel for reduction(+:axial_bins) schedule(dynamic)
//    #endif

    ViewSegmentNumbers vs_num;
    for (vs_num.segment_num()= this->proj_data_info_cyl_noarc_cor_sptr->get_min_segment_num();
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


            this->get_jacobian_for_view_segment_number(gradient_image_array,est_data,vs_num,compute_gradient,isgradient_mu);

            bin_counter +=
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_axial_poss(vs_num.segment_num()) *
            this->proj_data_info_cyl_noarc_cor_sptr->get_num_tangential_poss();
            //info(boost::format("ScatterSimulator: %d / %d") % bin_counter% total_bins);

            std::cout<< bin_counter << " / "<< total_bins <<std::endl;

        }
    }
return est_data;
}

void
SingleScatterLikelihoodAndGradient::
get_jacobian_for_view_segment_number(std::vector<VoxelsOnCartesianGrid<float> > &gradient_image_array, ProjData &est_data, const ViewSegmentNumbers& vs_num, const bool compute_gradient,const bool isgradient_mu)
{

    Viewgram<float> v_est = est_data.get_empty_viewgram(vs_num.view_num(), vs_num.segment_num());
    get_jacobian_for_viewgram(v_est,gradient_image_array, compute_gradient,isgradient_mu);
    est_data.set_viewgram(v_est);

}

void
SingleScatterLikelihoodAndGradient::
get_jacobian_for_viewgram(Viewgram<float>& v_est,std::vector<VoxelsOnCartesianGrid<float> > &gradient_image_array,const bool compute_gradient, const bool isgradient_mu)
{

    const ViewSegmentNumbers vs_num(v_est.get_view_num(),v_est.get_segment_num());

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
        if(gradient_image_array.size()!=static_cast<int>(all_bins.size())*this->output_proj_data_sptr->get_num_views())
            error("SIZE is %d , but it should be %d",gradient_image_array.size(),static_cast<int>(all_bins.size()));


    #ifdef STIR_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
       for (int i = 0; i < static_cast<int>(all_bins.size()); ++i)
       {   // this needs to be defined inside here to be thread-safe
           VoxelsOnCartesianGrid<float> tmp_gradient_image(gradient_image_array[0]);
           tmp_gradient_image.fill(0);

           const Bin bin = all_bins[i];
           const double y = L_G_estimate(tmp_gradient_image,bin,compute_gradient,isgradient_mu);

           v_est[bin.axial_pos_num()][bin.tangential_pos_num()] = static_cast<float>(y); //this is passed as reference and filled in the loop
           gradient_image_array[i] += tmp_gradient_image;
       }

}

ProjDataInMemory
SingleScatterLikelihoodAndGradient::
get_ratio(const ProjData& projdata,const ProjData& norm,const ProjData &add_projdata, const ProjData &est_projdata, std::vector<float> &ratio_vector)
{

    ProjDataInMemory ratio_HR(projdata); ratio_HR.fill(0);
    ProjDataInMemory est_projdata_HR(projdata); est_projdata_HR.fill(0);
    ProjDataInMemory ratio_LR(est_projdata); ratio_LR.fill(0);

    if((est_projdata.get_num_views()!=projdata.get_num_views())||(est_projdata.get_num_tangential_poss()!=projdata.get_num_tangential_poss()))
    ScatterEstimation::pull_scatter_estimate(est_projdata_HR,projdata,est_projdata,norm,true);
    else
    est_projdata_HR.fill(est_projdata);

    Bin bin;
    {
        for (bin.segment_num()=projdata.get_min_segment_num(); bin.segment_num()<=projdata.get_max_segment_num(); ++bin.segment_num())
            for (bin.axial_pos_num() =  projdata.get_min_axial_pos_num(bin.segment_num()); bin.axial_pos_num()<=projdata.get_max_axial_pos_num(bin.segment_num()); ++bin.axial_pos_num())
            {
                Sinogram<float> sino = projdata.get_sinogram(bin.axial_pos_num(),bin.segment_num());
                Sinogram<float> add_sino = add_projdata.get_sinogram(bin.axial_pos_num(),bin.segment_num());
                Sinogram<float> est_sino = est_projdata_HR.get_sinogram(bin.axial_pos_num(),bin.segment_num());
                Sinogram<float> ratio_sino = ratio_HR.get_empty_sinogram(bin.axial_pos_num(),bin.segment_num());

                for (bin.view_num()=sino.get_min_view_num();
                     bin.view_num()<=sino.get_max_view_num();
                     ++bin.view_num())
                {
                    for (bin.tangential_pos_num()=  sino.get_min_tangential_pos_num(); bin.tangential_pos_num()<= sino.get_max_tangential_pos_num();  ++bin.tangential_pos_num())
                        if(est_sino[bin.axial_pos_num()][bin.tangential_pos_num()]==0)
                        ratio_sino[bin.view_num()][bin.tangential_pos_num()] = 0;
                        else
                        ratio_sino[bin.view_num()][bin.tangential_pos_num()] = sino[bin.axial_pos_num()][bin.tangential_pos_num()]/(est_sino[bin.axial_pos_num()][bin.tangential_pos_num()]+add_sino[bin.axial_pos_num()][bin.tangential_pos_num()])-1;
                        ratio_HR.set_sinogram(ratio_sino);

                 }

              }
        }

    if((est_projdata.get_num_views()!=projdata.get_num_views())||(est_projdata.get_num_tangential_poss()!=projdata.get_num_tangential_poss()))
    ScatterEstimation::push_scatter_estimate(ratio_LR,est_projdata,ratio_HR,norm,true);
    else
    ratio_LR.fill(ratio_HR);

    ViewSegmentNumbers vs_num;
    int counter = 0;

    for (vs_num.segment_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_segment_num(); vs_num.segment_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_segment_num(); ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_view_num();vs_num.view_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_view_num(); ++vs_num.view_num())
        {

            Viewgram<float> viewgram = ratio_LR.get_viewgram(vs_num.view_num(), vs_num.segment_num(),false);
            const ViewSegmentNumbers vs_num(viewgram.get_view_num(),viewgram.get_segment_num());
            std::vector<Bin> all_bins;
            {
                Bin bin(vs_num.segment_num(), vs_num.view_num(), 0, 0);
                for (bin.axial_pos_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_axial_pos_num(bin.segment_num());  bin.axial_pos_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_axial_pos_num(bin.segment_num());  ++bin.axial_pos_num())
                {
                    for (bin.tangential_pos_num() = this->proj_data_info_cyl_noarc_cor_sptr->get_min_tangential_pos_num(); bin.tangential_pos_num() <= this->proj_data_info_cyl_noarc_cor_sptr->get_max_tangential_pos_num(); ++bin.tangential_pos_num())
                    {
                        all_bins.push_back(bin);
                    }
                }
            }

            for (int i = 0; i < static_cast<int>(all_bins.size()); ++i)
            {
              ++ counter;

              const Bin bin = all_bins[i];

              ratio_vector[i] = viewgram[bin.axial_pos_num()][bin.tangential_pos_num()];
            }

        }
    }

    if(ratio_vector.size()!=counter)
        error("SIZE is %d , but it should be %d",ratio_vector.size(),counter);
return est_projdata_HR;//
}
END_NAMESPACE_STIR

