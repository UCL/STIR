//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjMatrixByBinSinglePhoton

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/



#include "local/stir/recon_buildblock/ProjMatrixByBinSinglePhoton.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Coordinate3D.h"
#include <algorithm>
#include <math.h>

START_NAMESPACE_STIR


const char * const 
ProjMatrixByBinSinglePhoton::registered_name =
  "Single Photon";

ProjMatrixByBinSinglePhoton::
ProjMatrixByBinSinglePhoton()
{
  set_defaults();
}

void 
ProjMatrixByBinSinglePhoton::initialise_keymap()
{
  ProjMatrixByBin::initialise_keymap();
  parser.add_start_key("Single Photon Matrix Parameters");
  parser.add_stop_key("End Single Photon Matrix Parameters");
}


void
ProjMatrixByBinSinglePhoton::set_defaults()
{
  ProjMatrixByBin::set_defaults();
}


void
ProjMatrixByBinSinglePhoton::
set_up(		 
       const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr  
       )
{
  proj_data_info_ptr= proj_data_info_ptr_v; 
  if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(proj_data_info_ptr.get()) == 0)
    error("Single Photm projection matrix can handle on non-arccorrected data\n");

  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByBinSinglePhoton initialised with a wrong type of DiscretisedDensity\n");

 
  image_info_ptr->get_regular_range(min_index, max_index);

  if (min_index[1]!=0 || max_index[1]!=0)
    error("Image should have only 1 plane\n");
  if (max_index[2]-min_index[2]+1 != proj_data_info_ptr->get_scanner_ptr()->get_num_rings())
    error("Image should have y-dimension equal to the number of rings\n");
  if (max_index[3]-min_index[3]+1 != proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring())
    error("Image should have x-dimension equal to the number of detectors per ring\n");
  symmetries_ptr
    .reset(new TrivialDataSymmetriesForBins(proj_data_info_ptr));
  
};


void 
ProjMatrixByBinSinglePhoton::
calculate_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  const Bin bin = lor.get_bin();

  assert(lor.size() == 0);

  vector<DetectionPositionPair<> > det_pos_pairs;
  static_cast<ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr).
    get_all_det_pos_pairs_for_bin(det_pos_pairs, bin);
  for (std::vector<DetectionPositionPair<> >::const_iterator det_pos_pair_iter = det_pos_pairs.begin();
       det_pos_pair_iter != det_pos_pairs.end();
       ++det_pos_pair_iter)
    {
      lor.push_back(ProjMatrixElemsForOneBin::
		    value_type(Coordinate3D<int>(0,
						 det_pos_pair_iter->pos1().axial_coord() + min_index[2],
						 det_pos_pair_iter->pos1().tangential_coord() + min_index[3]),1)); 
      lor.push_back(ProjMatrixElemsForOneBin::
		    value_type(Coordinate3D<int>(0,
						 det_pos_pair_iter->pos2().axial_coord() + min_index[2],
						 det_pos_pair_iter->pos2().tangential_coord() + min_index[3]),1)); 
    }
}

         

END_NAMESPACE_STIR

