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


#include "tomo/recon_buildblock/BinNormalisationFromProjData.h"
#include "ProjData.h"
#include "shared_ptr.h"
#include "RelatedViewgrams.h"

START_NAMESPACE_TOMO

const char * const 
BinNormalisationFromProjData::registered_name = "From ProjData"; 

BinNormalisationFromProjData::
BinNormalisationFromProjData()
{
  set_defaults();
}

BinNormalisationFromProjData::
BinNormalisationFromProjData(const string& filename)
    : norm_proj_data_ptr(ProjData::read_from_file(filename.c_str()))
  {}

BinNormalisationFromProjData::
BinNormalisationFromProjData(const shared_ptr<ProjData>& norm_proj_data_ptr)
    : norm_proj_data_ptr(norm_proj_data_ptr)
  {}

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

  
END_NAMESPACE_TOMO

