/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2014, 2016, University College London
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

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Sanida Mustafovic
*/

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h" 
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h" 
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjData.h"
#include "stir/listmode/CListRecord.h"
#include "stir/Viewgram.h"
#include "stir/info.h"
#include <boost/format.hpp>

#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/recon_array_functions.h"

#include <iostream>
#include <algorithm>
#include <sstream>
#include "stir/stream.h"

#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

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
  this->additive_proj_data_sptr.reset();
  this->additive_projection_data_filename ="0"; 
  this->max_ring_difference_num_to_process =-1;
  this->PM_sptr.reset(new  ProjMatrixByBinUsingRayTracing());

  this->normalisation_sptr.reset(new TrivialBinNormalisation);
  this->do_time_frame = false;
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
 
  this->parser.add_key("num_events_to_store",&this->num_events_to_store);
} 
template <typename TargetT> 
int 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_num_subsets(const int new_num_subsets)
{
  this->num_subsets = new_num_subsets;
  return this->num_subsets;
}

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{
    assert(this->num_subsets>0);
        const DataSymmetriesForBins& symmetries =
                *this->PM_sptr->get_symmetries_ptr();

        Array<1,int> num_bins_in_subset(this->num_subsets);
        num_bins_in_subset.fill(0);


        for (int subset_num=0; subset_num<this->num_subsets; ++subset_num)
        {
            for (int segment_num = -this->max_ring_difference_num_to_process;
                 segment_num <= this->max_ring_difference_num_to_process; ++segment_num)
            {
                for (int axial_num = this->proj_data_info_cyl_uncompressed_ptr->get_min_axial_pos_num(segment_num);
                     axial_num < this->proj_data_info_cyl_uncompressed_ptr->get_max_axial_pos_num(segment_num);
                     axial_num ++)
                {
                    // For debugging.
                    //                std::cout <<segment_num << " "<<  axial_num  << std::endl;

                    for (int tang_num= this->proj_data_info_cyl_uncompressed_ptr->get_min_tangential_pos_num();
                         tang_num < this->proj_data_info_cyl_uncompressed_ptr->get_max_tangential_pos_num();
                         tang_num ++ )
                    {
                        for(int view_num = this->proj_data_info_cyl_uncompressed_ptr->get_min_view_num() + subset_num;
                            view_num <= this->proj_data_info_cyl_uncompressed_ptr->get_max_view_num();
                            view_num += this->num_subsets)
                        {
                            const Bin tmp_bin(segment_num,
                                              view_num,
                                              axial_num,
                                              tang_num, 1);

                            if (!this->PM_sptr->get_symmetries_ptr()->is_basic(tmp_bin) )
                                continue;

                            num_bins_in_subset[subset_num] +=
                                    symmetries.num_related_bins(tmp_bin);

                        }
                    }
                }
            }
        }

        for (int subset_num=1; subset_num<this->num_subsets; ++subset_num)
        {
            if(num_bins_in_subset[subset_num] != num_bins_in_subset[0])
            {
                std::stringstream str(warning_message);
                str <<"Number of subsets is such that subsets will be very unbalanced.\n"
                   << "Number of Bins in each subset would be:\n"
                   << num_bins_in_subset
                   << "\nEither reduce the number of symmetries used by the projector, or\n"
                      "change the number of subsets. It usually should be a divisor of\n"
                   << this->proj_data_info_cyl_uncompressed_ptr->get_num_views()
                   << "/4 (or if that's not an integer, a divisor of "
                   << this->proj_data_info_cyl_uncompressed_ptr->get_num_views()
                   << "/2 or "
                   << this->proj_data_info_cyl_uncompressed_ptr->get_num_views()
                   << ").\n";
                warning_message = str.str();
                return false;
            }
        }
        return true;
}

template <typename TargetT>  
Succeeded 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_up_before_sensitivity(shared_ptr <TargetT > const& target_sptr) 
{ 
#ifdef STIR_MPI
    //broadcast objective_function (100=PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin)
    distributed::send_int_value(100, -1);
#endif


    // set projector to be used for the calculations
    this->PM_sptr->set_up(this->proj_data_info_cyl_uncompressed_ptr->create_shared_clone(),target_sptr);

    shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingProjMatrixByBin(this->PM_sptr));
    shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingProjMatrixByBin(this->PM_sptr));

    this->projector_pair_ptr->set_up(this->proj_data_info_cyl_uncompressed_ptr->create_shared_clone(),target_sptr);

    this->projector_pair_ptr.reset(
                   new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));

    if (is_null_ptr(this->normalisation_sptr))
    {
        warning("Invalid normalisation object");
        return Succeeded::no;
    }

    if (this->normalisation_sptr->set_up(
                this->proj_data_info_cyl_uncompressed_ptr->create_shared_clone()) == Succeeded::no)
        return Succeeded::no;

    if (this->current_frame_num<=0)
    {
        warning("frame_num should be >= 1");
        return Succeeded::no;
    }

    if (this->current_frame_num > this->frame_defs.get_num_frames())
        {
            warning("frame_num is %d, but should be less than the number of frames %d.",
                    this->current_frame_num, this->frame_defs.get_num_frames());
            return Succeeded::no;
        }

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
  shared_ptr<Scanner> scanner_sptr(new Scanner(*this->list_mode_data_sptr->get_scanner_ptr()));


  if (this->max_ring_difference_num_to_process == -1)
    {
      this->max_ring_difference_num_to_process = 
        scanner_sptr->get_num_rings()-1;
    }


  if (this->additive_projection_data_filename != "0")
    {
      info(boost::format("Reading additive projdata data '%1%'")
           % additive_projection_data_filename  );
      shared_ptr <ProjData> temp_additive_proj_data_sptr =
        ProjData::read_from_file(this->additive_projection_data_filename);
      this->additive_proj_data_sptr.reset(new ProjDataInMemory(* temp_additive_proj_data_sptr));
    }


  this->proj_data_info_cyl_uncompressed_ptr.reset(
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
    ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                  1, this->max_ring_difference_num_to_process,
                  scanner_sptr->get_num_detectors_per_ring()/2,
                  scanner_sptr->get_max_num_non_arccorrected_bins(),
                  false)));

  if ( this->normalisation_sptr->set_up(proj_data_info_cyl_uncompressed_ptr)
   != Succeeded::yes)
  {
warning("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin: "
      "set-up of pre-normalisation failed\n");
return true;
    }



   return false; 

} 
 
template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{
//    std::cout << "!Nere " <<std::endl;
//    const int min_segment_num = -this->max_ring_difference_num_to_process;
//    const int max_segment_num = this->max_ring_difference_num_to_process;

//    std::cout << min_segment_num<<  " "<< max_segment_num <<std::endl;

//    for (int segment_num = min_segment_num; segment_num <= max_segment_num; ++segment_num)
//    {
//        std::cout << "here " <<  segment_num <<std::endl;
//        for (int axial_num = this->proj_data_info_cyl_uncompressed_ptr->get_min_axial_pos_num(segment_num);
//             axial_num < this->proj_data_info_cyl_uncompressed_ptr->get_max_axial_pos_num(segment_num);
//             axial_num ++)
//        {
//            // For debugging.
//            std::cout <<segment_num << " "<<  axial_num  << std::endl;

//            for (int tang_num= this->proj_data_info_cyl_uncompressed_ptr->get_min_tangential_pos_num();
//                 tang_num < this->proj_data_info_cyl_uncompressed_ptr->get_max_tangential_pos_num();
//                 tang_num ++ )
//            {

//                for(int view_num = this->proj_data_info_cyl_uncompressed_ptr->get_min_view_num() + subset_num;
//                    view_num <= this->proj_data_info_cyl_uncompressed_ptr->get_max_view_num();
//                    view_num += this->num_subsets)
//                {
//                    Bin tmp_bin(segment_num,
//                                view_num,
//                                axial_num,
//                                tang_num, 1.f);

//                    if (!this->PM_sptr->get_symmetries_ptr()->is_basic(tmp_bin) )
//                        continue;

//                    this->add_projmatrix_to_sensitivity(sensitivity, tmp_bin);
//                }
//            }
//        }
//    }



    const int min_segment_num = -this->max_ring_difference_num_to_process;
    const int max_segment_num = this->max_ring_difference_num_to_process;

    // warning: has to be same as subset scheme used as in distributable_computation
    for (int segment_num = min_segment_num; segment_num <= max_segment_num; ++segment_num)
    {
          //CPUTimer timer;
          //timer.start();

      for (int view = this->proj_data_info_cyl_uncompressed_ptr->get_min_view_num() + subset_num;
          view <= this->proj_data_info_cyl_uncompressed_ptr->get_max_view_num();
          view += this->num_subsets)
      {
        const ViewSegmentNumbers view_segment_num(view, segment_num);

        if (! this->projector_pair_ptr->get_symmetries_used()->is_basic(view_segment_num))
          continue;
        this->add_view_seg_to_sensitivity(sensitivity, view_segment_num);
      }
        //    cerr<<timer.value()<<endl;
    }
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
add_view_seg_to_sensitivity(TargetT& sensitivity, const ViewSegmentNumbers& view_seg_nums) const
{
    shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_used
            (this->projector_pair_ptr->get_symmetries_used()->clone());

  RelatedViewgrams<float> viewgrams =
    this->proj_data_info_cyl_uncompressed_ptr->get_empty_related_viewgrams(view_seg_nums,symmetries_used);

  viewgrams.fill(1.F);
  // find efficiencies
  {
    const double start_frame = this->frame_defs.get_start_time(this->current_frame_num);
    const double end_frame = this->frame_defs.get_end_time(this->current_frame_num);
    this->normalisation_sptr->undo(viewgrams,start_frame,end_frame);
  }
  // backproject
  {
    const int range_to_zero =
      view_seg_nums.segment_num() == 0
      ? 1 : 0;
    const int min_ax_pos_num =
      viewgrams.get_min_axial_pos_num() + range_to_zero;
    const int max_ax_pos_num =
       viewgrams.get_max_axial_pos_num() - range_to_zero;

    this->projector_pair_ptr->get_back_projector_sptr()->
      back_project(sensitivity, viewgrams,
                   min_ax_pos_num, max_ax_pos_num);
  }

}


template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
add_projmatrix_to_sensitivity(TargetT& sensitivity,  Bin & this_basic_bin) const
{
    std::vector<Bin>  r_bins;
    ProjMatrixElemsForOneBin elem_row;

    this->PM_sptr->get_symmetries_ptr()->get_related_bins(r_bins, this_basic_bin);

    if (r_bins.size() == 0 )
        error("Something went wrong with the symmetries. Abort.");

    for (unsigned int i = 0; i < r_bins.size(); i++)
        r_bins[i].set_bin_value(1.0f);


    // find efficiencies
    {
        const double start_frame = this->frame_defs.get_start_time(this->current_frame_num);
        const double end_frame = this->frame_defs.get_end_time(this->current_frame_num);
        this->normalisation_sptr->undo(r_bins, start_frame,end_frame);
    }

    this->PM_sptr->get_proj_matrix_elems_for_one_bin(elem_row, this_basic_bin);
    //N.E: I had problems with RelatedBins thats why I use a std::vector<Bin> and
    // symmetries.
    elem_row.back_project(sensitivity, r_bins, this->PM_sptr->get_symmetries_sptr());
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

  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);

  ProjDataInfoCylindricalNoArcCorr* proj_data_no_arc_ptr =
          dynamic_cast<ProjDataInfoCylindricalNoArcCorr *> ( this->proj_data_info_cyl_uncompressed_ptr.get());

  const double start_time = this->frame_defs.get_start_time(this->current_frame_num);
  const double end_time = this->frame_defs.get_end_time(this->current_frame_num);

  long num_stored_events = 0;
  const float max_quotient = 10000.F;

  //go to the beginning of this frame
  //  list_mode_data_sptr->set_get_position(start_time);
  // TODO implement function that will do this for a random time
  this->list_mode_data_sptr->reset();
  double current_time = 0.;
  ProjMatrixElemsForOneBin proj_matrix_row; 

  shared_ptr<CListRecord> record_sptr = this->list_mode_data_sptr->get_empty_record_sptr(); 
  CListRecord& record = *record_sptr; 

  unsigned long int more_events =
          this->do_time_frame? 1 : this->num_events_to_store;

  while (more_events)//this->list_mode_data_sptr->get_next_record(record) == Succeeded::yes)
  { 

      if (this->list_mode_data_sptr->get_next_record(record) == Succeeded::no)
              {
                  info("End of file!");
                  break; //get out of while loop
              }

    if(record.is_time() && end_time > 0.01)
      {
        current_time = record.time().get_time_in_secs();
      }
    if (this->do_time_frame && current_time >= end_time)
      {
        break; // get out of while loop
      }
    if (current_time < start_time)
      continue;

    if (record.is_event() && record.event().is_prompt()) 
      { 
        Bin measured_bin; 
        measured_bin.set_bin_value(1.0f);

        record.event().get_bin(measured_bin, *proj_data_info_cyl_uncompressed_ptr); 
        // If more than 1 subsets, check if the current bin belongs to
        // the current.
        if (this->num_subsets > 1)
        {
            Bin basic_bin = measured_bin;
            if (!this->PM_sptr->get_symmetries_ptr()->is_basic(measured_bin) )
                this->PM_sptr->get_symmetries_ptr()->find_basic_bin(basic_bin);

            if (subset_num != static_cast<int>(basic_bin.view_num() % this->num_subsets))
            {
                continue;
            }
        }

        this->PM_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, measured_bin); 
        //in_the_range++;
        Bin fwd_bin; 
        fwd_bin.set_bin_value(0.0f);
        proj_matrix_row.forward_project(fwd_bin,current_estimate); 
        // additive sinogram 
        if (!is_null_ptr(this->additive_proj_data_sptr))
          {
            float add_value = this->additive_proj_data_sptr->get_bin_value(measured_bin);
            float value= fwd_bin.get_bin_value()+add_value;         
            fwd_bin.set_bin_value(value);
          }
        float  measured_div_fwd = 0.0f;

        if(!this->do_time_frame)
             more_events -=1 ;

         num_stored_events += 1;

         if (num_stored_events%100000L==0)
                         info( boost::format("Proccessed events: %1% ") % num_stored_events);

        if ( measured_bin.get_bin_value() <= max_quotient *fwd_bin.get_bin_value())
            measured_div_fwd = 1.0f /fwd_bin.get_bin_value();
        else
            continue;

        measured_bin.set_bin_value(measured_div_fwd);
        proj_matrix_row.back_project(gradient, measured_bin); 
         
      } 
  } 
    info(boost::format("Number of used events: %1%") % num_stored_events);
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<DiscretisedDensity<3,float> >;


END_NAMESPACE_STIR
