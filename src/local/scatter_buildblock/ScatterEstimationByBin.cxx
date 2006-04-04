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
  \brief Implementation of stir::ScatterEstimationByBin

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
#include "local/stir/ScatterEstimationByBin.h"
#include "local/stir/Scatter.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h" 
#include "stir/Bin.h"
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
  this->use_polarization = false;
  this->scatter_level = 10 ;
  this->energy_resolution = .22 ;
  this->reference_energy = 511.F;
  this->lower_energy_threshold = 350 ;
  this->upper_energy_threshold = 650 ;
  this->activity_image_filename = "";
  this->density_image_filename = "";
  this->density_image_for_scatter_points_filename = "";
  this->template_proj_data_filename = "";
  this->output_proj_data_filename = "";

  this->write_scatter_orders_in_separate_files = true;
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
  this->parser.add_key("use_polarization", &this->use_polarization);
  this->parser.add_key("scatter_level", &this->scatter_level);
  this->parser.add_key("energy_resolution", &this->energy_resolution);
  this->parser.add_key("lower_energy_threshold", &this->lower_energy_threshold);
  this->parser.add_key("upper_energy_threshold", &this->upper_energy_threshold);

  this->parser.add_key("activity_image_filename", &this->activity_image_filename);
  this->parser.add_key("density_image_filename", &this->density_image_filename);
  this->parser.add_key("density_image_for_scatter_points_filename", &this->density_image_for_scatter_points_filename);
  this->parser.add_key("template_proj_data_filename", &this->template_proj_data_filename);
  this->parser.add_key("output_filename_prefix", &this->output_proj_data_filename);
  
  this->parser.add_key("write_scatter_orders_in_separate_files", &this->write_scatter_orders_in_separate_files);
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

  output_proj_data_00_sptr = 0;
  output_proj_data_01_sptr = 0;
  output_proj_data_11_sptr = 0;
  output_proj_data_02_sptr = 0;

  if (write_scatter_orders_in_separate_files)
    {
      if (this->scatter_level%10 == 0)
	{	  
	  output_proj_data_00_sptr = 
	    new ProjDataInterfile(proj_data_info_sptr,this->output_proj_data_filename + "_0_0");
	}
      if(scatter_level==1||scatter_level==12||scatter_level==10||scatter_level==120)
	{	  
	  output_proj_data_01_sptr = 
	    new ProjDataInterfile(proj_data_info_sptr,this->output_proj_data_filename + "_0_1");
	}
      if(scatter_level==2||scatter_level==12||scatter_level==120)
	{	  
	  output_proj_data_11_sptr = 
	    new ProjDataInterfile(proj_data_info_sptr,this->output_proj_data_filename + "_1_1");
	  output_proj_data_02_sptr = 
	    new ProjDataInterfile(proj_data_info_sptr,this->output_proj_data_filename + "_0_2");
	}
    }

  return false;
}

ScatterEstimationByBin::
ScatterEstimationByBin()
{
  this->set_defaults();
}

unsigned 
ScatterEstimationByBin::
find_in_detection_points_vector(const CartesianCoordinate3D<float>& coord) const
{
  std::vector<CartesianCoordinate3D<float> >::const_iterator iter=
    std::find(detection_points_vector.begin(),
	      detection_points_vector.end(),
	      coord);
  if (iter != detection_points_vector.end())
    {
      return iter-detection_points_vector.begin();
    }
  else
    {
      if (detection_points_vector.size()==static_cast<std::size_t>(total_detectors))
	error("More detection points than we think there are!\n");

      detection_points_vector.push_back(coord);
      return detection_points_vector.size()-1;
    }
}

void
ScatterEstimationByBin::
find_detectors(unsigned& det_num_A, unsigned& det_num_B, const Bin& bin) const
{
  CartesianCoordinate3D<float> detector_coord_A, detector_coord_B;
  this->proj_data_info_ptr->
    find_cartesian_coordinates_of_detection(
					    detector_coord_A,detector_coord_B,bin);
  det_num_A =
    this->find_in_detection_points_vector(detector_coord_A + 
					  this->shift_detector_coordinates_to_origin);
  det_num_B =
    this->find_in_detection_points_vector(detector_coord_B + 
					  this->shift_detector_coordinates_to_origin);
}

Succeeded 
ScatterEstimationByBin::
process_data()
{		
  this->proj_data_info_ptr = 
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *> 
    (this->output_proj_data_sptr->get_proj_data_info_ptr());

  this->sample_scatter_points();

  
  // find final size of detection_points_vector
  total_detectors = 
    this->proj_data_info_ptr->get_scanner_ptr()->get_num_rings()*
    this->proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring ();
  // reserve space to avoid reallocation, but the actual size will grow dynamically
  detection_points_vector.reserve(total_detectors);

  initialise_cache_for_scattpoint_det();
  initialise_cache_for_scattpoints();

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
  Bin bin;
	
  /* ////////////////// SCATTER ESTIMATION TIME ////////////////
   */
  CPUTimer bin_timer;
  int bin_counter = 0;
  bin_timer.start();
  int axial_bins = 0 ;
  for (bin.segment_num()=this->proj_data_info_ptr->get_min_segment_num();
       bin.segment_num()<=this->proj_data_info_ptr->get_max_segment_num();
       ++bin.segment_num())	
    axial_bins += this->proj_data_info_ptr->get_num_axial_poss(bin.segment_num());
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

  for (bin.segment_num()=this->proj_data_info_ptr->get_min_segment_num();
       bin.segment_num()<=this->proj_data_info_ptr->get_max_segment_num();
       ++bin.segment_num())
    {
      for (bin.view_num()=this->proj_data_info_ptr->get_min_view_num();
	   bin.view_num()<=this->proj_data_info_ptr->get_max_view_num();
	   ++bin.view_num())
	{
	  Viewgram<float> viewgram_00 =
	    this->output_proj_data_sptr->get_empty_viewgram(bin.view_num(), bin.segment_num());
	  Viewgram<float> viewgram_01 = viewgram_00;
	  Viewgram<float> viewgram_11 = viewgram_00;
	  Viewgram<float> viewgram_02 = viewgram_00;
	  Viewgram<float> viewgram_total_scatter = viewgram_00;
	      
			
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

		  double no_scatter = 0;
		  double scatter_ratio_01 = 0;
		  double scatter_ratio_11 = 0;
		  double scatter_ratio_02 = 0;
				
		  if(this->scatter_level%10==0)
		    {
		      no_scatter = 
			scatter_estimate_for_none_scatter_point
			(det_num_A, det_num_B
			 );
		    }
		  if(this->scatter_level!= 0)
		    {
		      scatter_estimate_for_all_scatter_points
			(
			 scatter_ratio_01,
			 scatter_ratio_11,
			 scatter_ratio_02,
			 det_num_A, 
			 det_num_B
			 );
		    }
		  
		  viewgram_00[bin.axial_pos_num()][bin.tangential_pos_num()] =
		    static_cast<float>(no_scatter);
		  viewgram_01[bin.axial_pos_num()][bin.tangential_pos_num()] =
		    static_cast<float>(scatter_ratio_01);
		  viewgram_11[bin.axial_pos_num()][bin.tangential_pos_num()] =
		    static_cast<float>(scatter_ratio_11);		      
		  viewgram_02[bin.axial_pos_num()][bin.tangential_pos_num()] =
		    static_cast<float>(scatter_ratio_02);
		  const double total_scatter_this_bin = 
		    scatter_ratio_01 + scatter_ratio_11 + scatter_ratio_02;
		  viewgram_total_scatter[bin.axial_pos_num()][bin.tangential_pos_num()] =
		    static_cast<float>(total_scatter_this_bin);

		  total_scatter += total_scatter_this_bin;

		  ++bin_counter;
		}
	    } // end loop over axial_pos

	  this->output_proj_data_sptr->set_viewgram(viewgram_total_scatter);
	  if (!is_null_ptr(this->output_proj_data_00_sptr))
	    this->output_proj_data_00_sptr->set_viewgram(viewgram_00);
	  if (!is_null_ptr(this->output_proj_data_01_sptr))
	    this->output_proj_data_01_sptr->set_viewgram(viewgram_01);
	  if (!is_null_ptr(this->output_proj_data_11_sptr))
	    this->output_proj_data_11_sptr->set_viewgram(viewgram_11);
	  if (!is_null_ptr(this->output_proj_data_02_sptr))
	    this->output_proj_data_02_sptr->set_viewgram(viewgram_02);
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
		
