//
//
/*
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
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
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisationFromProjData

  \author Kris Thielemans
*/

#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Succeeded.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>

START_NAMESPACE_STIR

const char * const 
BinNormalisationFromProjData::registered_name = "From ProjData"; 


void 
BinNormalisationFromProjData::set_defaults()
{
  normalisation_projdata_filename = "";
}

void 
BinNormalisationFromProjData::
initialise_keymap()
{
  parser.add_start_key("Bin Normalisation From ProjData");
  parser.add_key("normalisation_projdata_filename", &normalisation_projdata_filename);
  parser.add_stop_key("End Bin Normalisation From ProjData");
}

bool 
BinNormalisationFromProjData::
post_processing()
{
  norm_proj_data_ptr = ProjData::read_from_file(normalisation_projdata_filename);
  return false;
}

BinNormalisationFromProjData::
BinNormalisationFromProjData()
{
  set_defaults();
}

BinNormalisationFromProjData::
BinNormalisationFromProjData(const std::string& filename)
    : norm_proj_data_ptr(ProjData::read_from_file(filename))
  {}

BinNormalisationFromProjData::
BinNormalisationFromProjData(const shared_ptr<ProjData>& norm_proj_data_ptr)
    : norm_proj_data_ptr(norm_proj_data_ptr)
  {}

Succeeded 
BinNormalisationFromProjData::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
{
  BinNormalisation::set_up(proj_data_info_ptr);

  if (*(norm_proj_data_ptr->get_proj_data_info_ptr()) == *proj_data_info_ptr)
    return Succeeded::yes;
  else
  {
    const ProjDataInfo& norm_proj = *(norm_proj_data_ptr->get_proj_data_info_ptr());
    const ProjDataInfo& proj = *proj_data_info_ptr;
    bool ok = 
      (norm_proj >= proj) &&
      (norm_proj.get_min_tangential_pos_num() ==proj.get_min_tangential_pos_num())&&
      (norm_proj.get_max_tangential_pos_num() ==proj.get_max_tangential_pos_num());
    
    for (int segment_num=proj.get_min_segment_num();
	 ok && segment_num<=proj.get_max_segment_num();
	 ++segment_num)
      {
	ok = 
	  norm_proj.get_min_axial_pos_num(segment_num) == proj.get_min_axial_pos_num(segment_num) &&
	  norm_proj.get_max_axial_pos_num(segment_num) == proj.get_max_axial_pos_num(segment_num);
      }
    if (ok)
      return Succeeded::yes;
    else
      {
	warning(boost::format("BinNormalisationFromProjData: incompatible projection data:\nNorm projdata info:\n%s\nEmission projdata info:\n%s\n--- (end of incompatible projection data info)---\n")
		% norm_proj.parameter_info()
		% proj.parameter_info());
	return Succeeded::no;
      }
  }
}

bool 
BinNormalisationFromProjData::
is_trivial() const
{
  // check if all data is 1 (up to a tolerance of 1e-4)
  for (int segment_num = this->norm_proj_data_ptr->get_min_segment_num(); 
       segment_num <= this->norm_proj_data_ptr->get_max_segment_num(); 
       ++segment_num)
    {
      for (int view_num = this->norm_proj_data_ptr->get_min_view_num(); 
           view_num <= this->norm_proj_data_ptr->get_max_view_num(); 
           ++view_num)
        {
          const Viewgram<float> viewgram =
            this->norm_proj_data_ptr->get_viewgram(view_num, segment_num);
          if (fabs(viewgram.find_min()-1)>.0001 || fabs(viewgram.find_max()-1)>.0001)
            return false; // return from function as we know not all data is 1
        }
    }
  // if we get here. they were all 1
  return true;
}

void 
BinNormalisationFromProjData::apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
  {
    this->check(*viewgrams.get_proj_data_info_sptr());
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());
    viewgrams *= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_sptr, false);
  }

void 
BinNormalisationFromProjData::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
  {
    this->check(*viewgrams.get_proj_data_info_sptr());
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());
    viewgrams /= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_sptr, false);

  }

float 
BinNormalisationFromProjData::get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const
{
  //TODO
  error("BinNormalisationFromProjData::get_bin_efficiency is not implemented");
  return 1;

}
 
END_NAMESPACE_STIR

