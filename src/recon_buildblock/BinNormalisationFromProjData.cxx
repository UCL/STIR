//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Implementation for class BinNormalisationFromProjData

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Succeeded.h"

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
BinNormalisationFromProjData(const string& filename)
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
  if (*(norm_proj_data_ptr->get_proj_data_info_ptr()) == *proj_data_info_ptr)
    return Succeeded::yes;
  else
  {
    const ProjDataInfo& norm_proj = *(norm_proj_data_ptr->get_proj_data_info_ptr());
    const ProjDataInfo& proj = *proj_data_info_ptr;
    bool ok = 
      typeid(norm_proj) == typeid(proj) &&
      *norm_proj.get_scanner_ptr()== *(proj.get_scanner_ptr()) &&
      (norm_proj.get_min_view_num()==proj.get_min_view_num()) &&
      (norm_proj.get_max_view_num()==proj.get_max_view_num()) &&
      (norm_proj.get_min_tangential_pos_num() ==proj.get_min_tangential_pos_num())&&
      (norm_proj.get_max_tangential_pos_num() ==proj.get_max_tangential_pos_num()) &&
      norm_proj.get_min_segment_num() <= proj.get_min_segment_num() &&
      norm_proj.get_max_segment_num() <= proj.get_max_segment_num();
    
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
	warning("BinNormalisationFromProjData: incompatible projection data\n");
	return Succeeded::no;
      }
  }
}


void 
BinNormalisationFromProjData::apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams *= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone(), false);
  }

void 
BinNormalisationFromProjData::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams /= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone(), false);

  }

float 
BinNormalisationFromProjData::get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const
{
  //TODO
  return 1;

}
 
END_NAMESPACE_STIR

