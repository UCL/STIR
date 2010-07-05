//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
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
  \ingroup GeneralisedObjectiveFunction
  \brief Implementation of class 
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h" 
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h" 
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjData.h"
#include "stir/listmode/CListRecord.h"
#include "stir/Viewgram.h"
#ifdef STIR_MPI
#include "stir/recon_buildblock/distributed_functions.h"
#endif


#include <vector>
START_NAMESPACE_STIR

template<typename TargetT>
const char * const 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
registered_name = 
"PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin";
 
template <typename TargetT> 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>:: 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin() 
{ 
  this->set_defaults(); 
} 

template <typename TargetT> 
void  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_defaults() 
{ 
  base_type::set_defaults();
  this->additive_proj_data_sptr =NULL; 
  this->additive_projection_data_filename ="0"; 
  this->max_ring_difference_num_to_process =-1;
  this->PM_sptr =  
    new  ProjMatrixByBinUsingRayTracing(); 
} 
 
template <typename TargetT> 
void  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
initialise_keymap() 
{ 
  base_type::initialise_keymap(); 
  this->parser.add_start_key("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin Parameters"); 
  this->parser.add_stop_key("End PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin Parameters"); 
  this->parser.add_key("max ring difference num to process", &this->max_ring_difference_num_to_process);
  this->parser.add_parsing_key("Matrix type", &this->PM_sptr); 
  this->parser.add_key("additive sinogram",&this->additive_projection_data_filename); 
 
   
} 
template <typename TargetT> 
int 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_num_subsets(const int new_num_subsets)
{
  if (new_num_subsets!=1)
    warning("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin currently only supports 1 subset");
  this->num_subsets = 1;
  return this->num_subsets;
}

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
actual_subsets_are_approximately_balanced(string&) const
{
  return true; 
}

template <typename TargetT>  
Succeeded 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::set_up(shared_ptr <TargetT > const& target_sptr) 
{ 
#ifdef STIR_MPI
        //broadcast objective_function (100=PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin)
        distributed::send_int_value(100, -1);
#endif
        
 
  if (base_type::set_up(target_sptr) != Succeeded::yes) 
    return Succeeded::no; 

  // set projector to be used for the calculations    
  this->PM_sptr->set_up(this->proj_data_info_cyl_uncompressed_ptr->clone(),target_sptr); 

  return Succeeded::yes; 
} 
 
 
template <typename TargetT>  
bool  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::post_processing() 
{ 

  if (base_type::post_processing() == true) 
    return true; 
 
#if 1
  if (is_null_ptr(this->PM_sptr)) 

    { warning("You need to specify a projection matrix"); return true; } 

#else
  if(is_null_ptr(this->projector_pair_sptr->get_forward_projector_sptr()))
    {
      warning("No valid forward projector is defined"); return true;
    }

  if(is_null_ptr(this->projector_pair_sptr->get_back_projector_sptr()))
    {
      warning("No valid back projector is defined"); return true;
    }
#endif
  shared_ptr<Scanner> scanner_sptr = new Scanner(*this->list_mode_data_sptr->get_scanner_ptr());

  if (this->max_ring_difference_num_to_process == -1)
    {
      this->max_ring_difference_num_to_process = 
        scanner_sptr->get_num_rings()-1;
    }

     
  if (this->additive_projection_data_filename != "0") 
    { 
      cerr << "\nReading additive projdata data " 
           << additive_projection_data_filename  
           << endl; 
      shared_ptr <ProjData> temp_additive_proj_data_sptr =  
        ProjData::read_from_file(this->additive_projection_data_filename); 
      this->additive_proj_data_sptr = new ProjDataInMemory(* temp_additive_proj_data_sptr);
    } 
  

  this->proj_data_info_cyl_uncompressed_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
    ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
                  1, this->max_ring_difference_num_to_process,
                  scanner_sptr->get_num_detectors_per_ring()/2,
                  scanner_sptr->get_default_num_arccorrected_bins(), 
                  false));
   return false; 

} 
 
 
 
template <typename TargetT> 
TargetT * 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>:: 
construct_target_ptr() const 
{ 

 return 
      new VoxelsOnCartesianGrid<float> (*this->proj_data_info_cyl_uncompressed_ptr, 
                                        static_cast<float>(this->zoom), 
                                        CartesianCoordinate3D<float>(static_cast<float>(this->Zoffset), 
                                                                     static_cast<float>(this->Yoffset), 
                                                                     static_cast<float>(this->Xoffset)), 
                                        CartesianCoordinate3D<int>(this->output_image_size_z, 
                                                                   this->output_image_size_xy, 
                                                                   this->output_image_size_xy) 
                                       ); 

} 
 
template <typename TargetT> 
void 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>:: 
compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,  
                                         const TargetT &current_estimate,  
                                         const int subset_num) 
{ 

  const double start_time = this->frame_defs.get_start_time(this->current_frame_num);
  const double end_time = this->frame_defs.get_end_time(this->current_frame_num);
  //go to the beginning of this frame
  //  list_mode_data_sptr->set_get_position(start_time);
  // TODO implement function that will do this for a random time
  this->list_mode_data_sptr->reset();
  double current_time = start_time;
  ProjMatrixElemsForOneBin proj_matrix_row; 
  shared_ptr<CListRecord> record_sptr = this->list_mode_data_sptr->get_empty_record_sptr(); 
  CListRecord& record = *record_sptr; 
  //  int count_of_events=0;
  //int in_the_range =0;
  while (this->list_mode_data_sptr->get_next_record(record) == Succeeded::yes) 
  { 
    //count_of_events++;
     if(record.is_time())
      {
       const double new_time = record.time().get_time_in_secs();
       if ( new_time >= end_time)
           break; // get out of while loop
       current_time = new_time;
      }
    else if (record.is_event() && record.event().is_prompt()) 
      { 
        Bin measured_bin; 
        record.event().get_bin(measured_bin, *proj_data_info_cyl_uncompressed_ptr); 
        if (measured_bin.get_bin_value() <= 0)
          continue;      
        this->PM_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, measured_bin); 
        //in_the_range++;
        Bin fwd_bin; 
        proj_matrix_row.forward_project(fwd_bin,current_estimate); 
        // additive sinogram 
        if (!is_null_ptr(this->additive_proj_data_sptr))
          {
            float add_value = this->additive_proj_data_sptr->get_bin_value(measured_bin);
            float value= fwd_bin.get_bin_value()+add_value;         
            fwd_bin.set_bin_value(value);
          }
        float  measured_div_fwd = measured_bin.get_bin_value()/fwd_bin.get_bin_value();
        measured_bin.set_bin_value(measured_div_fwd);
        proj_matrix_row.back_project(gradient, measured_bin); 
         
      } 
  }
  //  cerr << " The number_of_events " << count_of_events << "   ";
  //cerr << " The number of events proecessed "  << in_the_range << "  ";
    
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<DiscretisedDensity<3,float> >;


END_NAMESPACE_STIR
