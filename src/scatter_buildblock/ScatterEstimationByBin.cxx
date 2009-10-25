//
// $Id$
//
/*
  Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
#include "stir/scatter/ScatterEstimationByBin.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h" 
#include "stir/Bin.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/CPUTimer.h"
#include "stir/Viewgram.h"
#include "stir/is_null_ptr.h"
#include <fstream>


START_NAMESPACE_STIR

void
ScatterEstimationByBin::
set_defaults()
{
  this->attenuation_threshold =  0.01 ;
  this->random = true;
  this->use_cache = true;
  this->energy_resolution = .22 ;
  this->reference_energy = 511.F;
  this->lower_energy_threshold = 350 ;
  this->upper_energy_threshold = 650 ;
  this->activity_image_filename = "";
  this->density_image_filename = "";
  this->density_image_for_scatter_points_filename = "";
  this->template_proj_data_filename = "";
  this->output_proj_data_filename = "";
}

void
ScatterEstimationByBin::
initialise_keymap()
{
  this->parser.add_start_key("Scatter Estimation Parameters");
  this->parser.add_stop_key("end Scatter Estimation Parameters");
  this->parser.add_key("attenuation_threshold", &this->attenuation_threshold);
  this->parser.add_key("random", &this->random);

  this->parser.add_key("use_cache", &this->use_cache);
  this->parser.add_key("energy_resolution", &this->energy_resolution);
  this->parser.add_key("lower_energy_threshold", &this->lower_energy_threshold);
  this->parser.add_key("upper_energy_threshold", &this->upper_energy_threshold);

  this->parser.add_key("activity_image_filename", &this->activity_image_filename);
  this->parser.add_key("density_image_filename", &this->density_image_filename);
  this->parser.add_key("density_image_for_scatter_points_filename", &this->density_image_for_scatter_points_filename);
  this->parser.add_key("template_proj_data_filename", &this->template_proj_data_filename);
  this->parser.add_key("output_filename_prefix", &this->output_proj_data_filename);
}

bool
ScatterEstimationByBin::
post_processing()
{
  this->activity_image_sptr= 
    DiscretisedDensity<3,float>::read_from_file(this->activity_image_filename);
  this->density_image_sptr= 
    DiscretisedDensity<3,float>::read_from_file(this->density_image_filename);
  this->density_image_for_scatter_points_sptr= 
    DiscretisedDensity<3,float>::read_from_file(this->density_image_for_scatter_points_filename);

  if (is_null_ptr(this->activity_image_sptr))
    {
      warning("Error reading activity image %s",
              this->activity_image_filename.c_str());
      return true;
    }
  if (is_null_ptr(this->density_image_for_scatter_points_sptr))
    {
      warning("Error reading density image %s",
              this->density_image_filename.c_str());
      return true;
    }
  if (is_null_ptr(this->density_image_for_scatter_points_sptr))
    {
      warning("Error reading density_for_scatter_points image %s",
              this->density_image_for_scatter_points_filename.c_str());
      return true;
    }
        
  warning("\nWARNING: Attenuation image data are supposed to be in units cm^-1\n"
          "\tReference: water has mu .096 cm^-1\n" 
          "\tMax in attenuation image: %g\n" ,
          this->density_image_sptr->find_max());
  shared_ptr<ProjData> template_proj_data_sptr = 
    ProjData::read_from_file(this->template_proj_data_filename);  
  const ProjDataInfoCylindricalNoArcCorr* proj_data_info_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(
                                                           template_proj_data_sptr->get_proj_data_info_ptr());
        
  if (is_null_ptr(proj_data_info_ptr))
    {     
      warning("ScatterEstimationByBin can only handle non-arccorrected data");
      return true;
    }


  shared_ptr<ProjDataInfo> proj_data_info_sptr = 
    proj_data_info_ptr->clone();

  output_proj_data_sptr = 
    new ProjDataInterfile(proj_data_info_sptr,this->output_proj_data_filename);



  // XXX should go to set_up
  {
  this->proj_data_info_ptr = 
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *> 
    (this->output_proj_data_sptr->get_proj_data_info_ptr());

  this->sample_scatter_points();

  
  // find final size of detection_points_vector
  this->total_detectors = 
    this->proj_data_info_ptr->get_scanner_ptr()->get_num_rings()*
    this->proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring ();
  // reserve space to avoid reallocation, but the actual size will grow dynamically
  detection_points_vector.reserve(total_detectors);

  this->initialise_cache_for_scattpoint_det();

#if 0
  {
    std::ofstream scatter_points_file("scatter_points.txt"); 
    if(!scatter_points_file)    
      warning("Cannot open scatter_points file.\n") ;                 
    else
      scatter_points_file << scatt_points_vector;
    std::cerr << scatt_points_vector.size() << " scatter points selected!" << std::endl;                                
  }
#endif

  }
  return false;
}

ScatterEstimationByBin::
ScatterEstimationByBin()
{
  this->set_defaults();
}

Succeeded 
ScatterEstimationByBin::
process_data()
{               
  ViewSegmentNumbers vs_num;
        
  /* ////////////////// SCATTER ESTIMATION TIME ////////////////
   */
  CPUTimer bin_timer;
  int bin_counter = 0;
  bin_timer.start();
  int axial_bins = 0 ;
  for (vs_num.segment_num()=this->proj_data_info_ptr->get_min_segment_num();
       vs_num.segment_num()<=this->proj_data_info_ptr->get_max_segment_num();
       ++vs_num.segment_num())  
    axial_bins += this->proj_data_info_ptr->get_num_axial_poss(vs_num.segment_num());
  const int total_bins = 
    this->proj_data_info_ptr->get_num_views() * axial_bins *
    this->proj_data_info_ptr->get_num_tangential_poss();

  /* ////////////////// end SCATTER ESTIMATION TIME ////////////////
   */
        
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
                                                                      detector_coord_A,detector_coord_B,Bin(0,0,0,0));
    assert(detector_coord_A.z()==0);
    assert(detector_coord_B.z()==0);
    // check that get_m refers to the middle of the scanner
    const float m_first =
      this->proj_data_info_ptr->get_m(Bin(0,0,this->proj_data_info_ptr->get_min_axial_pos_num(0),0));
    const float m_last =
      this->proj_data_info_ptr->get_m(Bin(0,0,this->proj_data_info_ptr->get_max_axial_pos_num(0),0));
    assert(fabs(m_last + m_first)<m_last*10E-4);
  }
#endif
  this->shift_detector_coordinates_to_origin =
    CartesianCoordinate3D<float>(this->proj_data_info_ptr->get_m(Bin(0,0,0,0)),0, 0);

  float total_scatter = 0 ;

  for (vs_num.segment_num()=this->proj_data_info_ptr->get_min_segment_num();
       vs_num.segment_num()<=this->proj_data_info_ptr->get_max_segment_num();
       ++vs_num.segment_num())
    {
      for (vs_num.view_num()=this->proj_data_info_ptr->get_min_view_num();
           vs_num.view_num()<=this->proj_data_info_ptr->get_max_view_num();
           ++vs_num.view_num())
        {
          total_scatter += this->process_data_for_view_segment_num(vs_num);
          bin_counter +=  
            this->proj_data_info_ptr->get_num_axial_poss(vs_num.segment_num()) *
            this->proj_data_info_ptr->get_num_tangential_poss();

          /* ////////////////// SCATTER ESTIMATION TIME ////////////////
           */
          {
            // TODO remove statics
            static double previous_timer = 0 ;          
            static int previous_bin_count = 0 ;
            std::cerr << bin_counter << " bins  Total time elapsed "
                      << bin_timer.value() << " sec \tTime remaining about "
                      << (bin_timer.value()-previous_timer)*
                         (total_bins-bin_counter)/
                         (bin_counter-previous_bin_count)/60
                      << " minutes"
                      << std::endl;                             
            previous_timer = bin_timer.value() ;
            previous_bin_count = bin_counter ;
          }
          /* ////////////////// end SCATTER ESTIMATION TIME ////////////////
           */
        }
    }
  bin_timer.stop();             
  this->write_log(bin_timer.value(), total_scatter);

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
ScatterEstimationByBin::
process_data_for_view_segment_num(const ViewSegmentNumbers& vs_num)
{
  Bin bin(vs_num.segment_num(), vs_num.view_num(), 0,0);
  double total_scatter = 0;
  Viewgram<float> viewgram =
    this->output_proj_data_sptr->get_empty_viewgram(bin.view_num(), bin.segment_num());       

                
  for (bin.axial_pos_num()=this->proj_data_info_ptr->get_min_axial_pos_num(bin.segment_num());
       bin.axial_pos_num()<=this->proj_data_info_ptr->get_max_axial_pos_num(bin.segment_num());
       ++bin.axial_pos_num())
    {
      for (bin.tangential_pos_num()=this->proj_data_info_ptr->get_min_tangential_pos_num();
           bin.tangential_pos_num()<=this->proj_data_info_ptr->get_max_tangential_pos_num();
           ++bin.tangential_pos_num())
        {  

          unsigned det_num_A = 0; // initialise to avoid compiler warnings
          unsigned det_num_B = 0;
          this->find_detectors(det_num_A, det_num_B, bin);

          const double scatter_ratio =
            scatter_estimate(det_num_A, det_num_B);
                  
          viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] =
            static_cast<float>(scatter_ratio);

          total_scatter += scatter_ratio;

        }
    } // end loop over axial_pos

  if (this->output_proj_data_sptr->set_viewgram(viewgram) == Succeeded::no)
    error("ScatterEstimationByBin: error writing viewgram");

  return static_cast<double>(viewgram.sum());
}


void
ScatterEstimationByBin::
write_log(const double simulation_time, 
          const float total_scatter)
{       

  std::string log_filename = 
    this->output_proj_data_filename + ".log";
  std::ofstream mystream(log_filename.c_str());
  if(!mystream)    
    {
      warning("Cannot open log file '%s'", log_filename.c_str()) ;
      return;
    }
  int axial_bins = 0 ;
  for (int segment_num=this->output_proj_data_sptr->get_min_segment_num();
       segment_num<=this->output_proj_data_sptr->get_max_segment_num();
       ++segment_num)   
    axial_bins += this->output_proj_data_sptr->get_num_axial_poss(segment_num); 
  const int total_bins = 
    this->output_proj_data_sptr->get_num_views() * axial_bins *
    this->output_proj_data_sptr->get_num_tangential_poss();     

  mystream << this->parameter_info()
           << "\nTotal simulation time elapsed: "                                 
           <<   simulation_time/60 
           << "\nTotal Scatter Points : " << scatt_points_vector.size() 
           << "\nTotal Scatter Counts : " << total_scatter 
           << "\nActivity image SIZE: " 
           << (*this->activity_image_sptr).size() << " * " 
           << (*this->activity_image_sptr)[0].size() << " * "  // TODO relies on 0 index
           << (*this->activity_image_sptr)[0][0].size()
           << "\nAttenuation image SIZE: " 
           << (*this->density_image_sptr).size() << " * "
           << (*this->density_image_sptr)[0].size() << " * " 
           << (*this->density_image_sptr)[0][0].size()
           << "\nTotal bins : " << total_bins << " = " 
           << this->output_proj_data_sptr->get_num_views()               
           << " view_bins * " 
           << axial_bins << " axial_bins * "
           << this->output_proj_data_sptr->get_num_tangential_poss() 
           << " tangential_bins\n"; 
}

END_NAMESPACE_STIR
                
