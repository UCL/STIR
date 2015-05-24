/*
  Copyright (C) 2004 -  2009 Hammersmith Imanet Ltd
  Copyright (C) 2013 University College London
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
*/
#include "stir/scatter/ScatterEstimationByBin.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h" 
#include "stir/Bin.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/Viewgram.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include "stir/error.h"
#include <fstream>
#include <boost/format.hpp>


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

  this->remove_cache_for_integrals_over_activity();
  this->remove_cache_for_integrals_over_attenuation();
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
  this->set_activity_image(this->activity_image_filename);
  this->set_density_image(this->density_image_filename);
  this->set_density_image_for_scatter_points(this->density_image_for_scatter_points_filename);
        
  info(boost::format("Attenuation image data are supposed to be in units cm^-1\n"
                     "\tReference: water has mu .096 cm^-1\n" 
                     "\tMax in attenuation image: %g\n") %
       this->density_image_sptr->find_max());

  this->set_template_proj_data_info(this->template_proj_data_filename);
  // create output (has to be AFTER set_template_proj_data_info)
  this->set_output_proj_data(this->output_proj_data_filename);

  return false;
}

ScatterEstimationByBin::
ScatterEstimationByBin()
{
  this->set_defaults();
}


/****************** functions to set images **********************/
void
ScatterEstimationByBin::
set_activity_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >& new_sptr)
{
  this->activity_image_sptr=new_sptr;
  this->remove_cache_for_integrals_over_activity();
}

void
ScatterEstimationByBin::
set_activity_image(const std::string& filename)  
{
  this->activity_image_filename=filename;
  this->activity_image_sptr= 
    read_from_file<DiscretisedDensity<3,float> >(filename);
  if (is_null_ptr(this->activity_image_sptr))
    {
      error(boost::format("Error reading activity image %s") %
            this->activity_image_filename);
    }
  this->set_activity_image_sptr(this->activity_image_sptr);
}


void
ScatterEstimationByBin::
set_density_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >& new_sptr)
{
  this->density_image_sptr=new_sptr;
  this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterEstimationByBin::
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
  this->set_density_image_sptr(this->density_image_sptr);
}

void
ScatterEstimationByBin::
set_density_image_for_scatter_points_sptr(const shared_ptr<DiscretisedDensity<3,float> >& new_sptr)
{
  this->density_image_for_scatter_points_sptr=new_sptr;
  this->sample_scatter_points();
  this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterEstimationByBin::
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
  this->set_density_image_for_scatter_points_sptr(this->density_image_for_scatter_points_sptr);
}


/****************** functions to set projection data **********************/

void
ScatterEstimationByBin::
set_template_proj_data_info_sptr(const shared_ptr<ProjDataInfo>& new_sptr)
{
  
  this->proj_data_info_ptr = dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(new_sptr->clone());

  if (is_null_ptr(this->proj_data_info_ptr))
    {     
      error("ScatterEstimationByBin can only handle non-arccorrected data");
    }
      
  // find final size of detection_points_vector
  this->total_detectors = 
    this->proj_data_info_ptr->get_scanner_ptr()->get_num_rings()*
    this->proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring ();
  // reserve space to avoid reallocation, but the actual size will grow dynamically
  this->detection_points_vector.reserve(total_detectors);

  // remove any cached values as they'd be incorrect if the sizes changes
  this->remove_cache_for_integrals_over_attenuation();
  this->remove_cache_for_integrals_over_activity();
}

void
ScatterEstimationByBin::
set_template_proj_data_info(const std::string& filename)
{
  this->template_proj_data_filename = filename;
  shared_ptr<ProjData> template_proj_data_sptr = 
    ProjData::read_from_file(this->template_proj_data_filename);  

  this->set_template_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone());
}

/*
void
ScatterEstimationByBin::
set_output_proj_data_sptr(const shared_ptr<ProjData>& new_sptr)
{
  this->output_proj_data_sptr = new_sptr;
}
*/

void
ScatterEstimationByBin::
set_output_proj_data(const std::string& filename)
{
  this->output_proj_data_filename = filename;
  // TODO get ExamInfo from image
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  this->output_proj_data_sptr.reset(new ProjDataInterfile(exam_info_sptr,
							  this->proj_data_info_ptr->create_shared_clone(),
							  this->output_proj_data_filename));
}

/****************** functions to compute scatter **********************/

Succeeded 
ScatterEstimationByBin::
process_data()
{               
  this->initialise_cache_for_scattpoint_det_integrals_over_attenuation();
  this->initialise_cache_for_scattpoint_det_integrals_over_activity();
 
  ViewSegmentNumbers vs_num;
        
  /* ////////////////// SCATTER ESTIMATION TIME ////////////////
   */
  CPUTimer bin_timer;
  bin_timer.start();
  // variables to report (remaining) time
  HighResWallClockTimer wall_clock_timer;
  double previous_timer = 0 ;          
  int previous_bin_count = 0 ;
  int bin_counter = 0;
  int axial_bins = 0 ;
  wall_clock_timer.start();

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

            wall_clock_timer.stop(); // must be stopped before getting the value
            info(boost::format("%1% bins  Total time elapsed %2% sec "
              "\tTime remaining about %3% minutes") 
              % bin_counter 
              % wall_clock_timer.value() 
              % ((wall_clock_timer.value()-previous_timer)
                *(total_bins-bin_counter)/(bin_counter-previous_bin_count)/60) );

            previous_timer = wall_clock_timer.value() ;
            previous_bin_count = bin_counter ;

            wall_clock_timer.start();
          }
          /* ////////////////// end SCATTER ESTIMATION TIME ////////////////
           */
        }
    }
  bin_timer.stop();
  wall_clock_timer.stop();
  this->write_log(wall_clock_timer.value(), total_scatter);

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
#ifdef STIR_OPENMP
#pragma omp parallel for firstprivate(bin) reduction(+:total_scatter) schedule(dynamic)
#endif
      for (int tang_pos_num=this->proj_data_info_ptr->get_min_tangential_pos_num();
           tang_pos_num<=this->proj_data_info_ptr->get_max_tangential_pos_num();
           ++tang_pos_num)
        {  
          bin.tangential_pos_num() = tang_pos_num;

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
           <<   simulation_time/60 << "min"
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
                
