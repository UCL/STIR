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
    warning("BinNormalisationFromProjData: incompatible projection data\n");
    return Succeeded::no;
  }
}


void 
BinNormalisationFromProjData::apply(RelatedViewgrams<float>& viewgrams) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams *= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone(), false);
  }

void 
BinNormalisationFromProjData::
undo(RelatedViewgrams<float>& viewgrams) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams /= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone(), false);

  }

  
END_NAMESPACE_STIR

