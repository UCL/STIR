//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief utilities for finding normalisation factors using an ML approach

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/ML_norm.h"

START_NAMESPACE_STIR


void make_det_pair_data(DetPairData& det_pair_data,
			const ProjData& proj_data,
			const int segment_num,
			const int ax_pos_num)
{
  const ProjDataInfo* proj_data_info_ptr =
    proj_data.get_proj_data_info_ptr();
  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr);

  const int num_detectors = 
    proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  det_pair_data.grow(IndexRange2D(0,num_detectors-1, 0,num_detectors-1));
  det_pair_data.fill(0);

  shared_ptr<Sinogram<float> > pos_sino_ptr =
    new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,segment_num));
  shared_ptr<Sinogram<float> > neg_sino_ptr;
  if (segment_num == 0)
    neg_sino_ptr = pos_sino_ptr;
  else
    neg_sino_ptr =
      new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,-segment_num));
  
    
  for (int view_num = 0; view_num < num_detectors/2; view_num++)
    for (int tang_pos_num = proj_data.get_min_tangential_pos_num();
	 tang_pos_num <= proj_data.get_max_tangential_pos_num();
	 ++tang_pos_num)
      {
	int det_num_a = 0;
	int det_num_b = 0;

	proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det_num_a, det_num_b, view_num, tang_pos_num);

	det_pair_data[det_num_a][det_num_b] =
	  (*pos_sino_ptr)[view_num][tang_pos_num];
	det_pair_data[det_num_b][det_num_a] =
	  (*neg_sino_ptr)[view_num][tang_pos_num];
      }
}

void set_det_pair_data(ProjData& proj_data,
                       const DetPairData& det_pair_data,
			const int segment_num,
			const int ax_pos_num)
{
  const ProjDataInfo* proj_data_info_ptr =
    proj_data.get_proj_data_info_ptr();
  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr);

  const int num_detectors = 
    proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  assert(det_pair_data.get_length() == num_detectors);

  shared_ptr<Sinogram<float> > pos_sino_ptr =
    new Sinogram<float>(proj_data.get_empty_sinogram(ax_pos_num,segment_num));
  shared_ptr<Sinogram<float> > neg_sino_ptr;
  if (segment_num != 0)
    neg_sino_ptr =
      new Sinogram<float>(proj_data.get_empty_sinogram(ax_pos_num,-segment_num));
  
    
  for (int view_num = 0; view_num < num_detectors/2; view_num++)
    for (int tang_pos_num = proj_data.get_min_tangential_pos_num();
	 tang_pos_num <= proj_data.get_max_tangential_pos_num();
	 ++tang_pos_num)
      {
	int det_num_a = 0;
	int det_num_b = 0;

	proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det_num_a, det_num_b, view_num, tang_pos_num);

	(*pos_sino_ptr)[view_num][tang_pos_num] =
          det_pair_data[det_num_a][det_num_b];
        if (segment_num!=0)
	  (*neg_sino_ptr)[view_num][tang_pos_num] =
            det_pair_data[det_num_b][det_num_a];
      }
  proj_data.set_sinogram(*pos_sino_ptr);
  if (segment_num != 0)
    proj_data.set_sinogram(*neg_sino_ptr);
}


void apply_block_norm(DetPairData& det_pair_data, const BlockData& block_data, const bool apply)
{
  const int num_detectors = det_pair_data.get_length();
  const int num_blocks = block_data.get_length();
  const int num_crystals_per_block = num_detectors/num_blocks;
  assert(num_blocks * num_crystals_per_block == num_detectors);
  
  for (int a = 0; a < num_detectors; ++a)
    for (int b = 0; b < num_detectors; ++b)      
      {
	// note: add 2*num_detectors to newb to avoid using mod with negative numbers
	if (det_pair_data[a][b] == 0)
	  continue;
        if (apply)
          det_pair_data[a][b] *=
	    block_data[a/num_crystals_per_block][b/num_crystals_per_block];
        else
          det_pair_data[a][b] /=
	    block_data[a/num_crystals_per_block][b/num_crystals_per_block];
      }
}

void apply_geo_norm(DetPairData& det_pair_data, const GeoData& geo_data, const bool apply)
{
  const int num_detectors = det_pair_data.get_length();
  const int num_crystals_per_block = geo_data.get_length()*2;

  for (int a = 0; a < num_detectors; ++a)
    for (int b = 0; b < num_detectors; ++b)      
      {
	if (det_pair_data[a][b] == 0)
	  continue;
        int newa = a % num_crystals_per_block;
	int newb = b - (a - newa); 
	if (newa > num_crystals_per_block - 1 - newa)
	  { 
	    newa = num_crystals_per_block - 1 - newa; 
	    newb = - newb + num_crystals_per_block - 1;
	  }
	// note: add 2*num_detectors to newb to avoid using mod with negative numbers
        if (apply)
          det_pair_data[a][b] *=
	    geo_data[newa][(2*num_detectors + newb)%num_detectors];
        else
          det_pair_data[a][b] /=
	    geo_data[newa][(2*num_detectors + newb)%num_detectors];
      }
}

void apply_efficiencies(DetPairData& det_pair_data, const Array<1,float>& efficiencies, const bool apply)
{
  const int num_detectors = det_pair_data.get_length();
  for (int det_num_a = 0; det_num_a < num_detectors; ++det_num_a)
    for (int det_num_b = 0; det_num_b < num_detectors; ++det_num_b)      
      {
	if (det_pair_data[det_num_a][det_num_b] == 0)
	  continue;
        if (apply)
	  det_pair_data[det_num_a][det_num_b] *=
	    efficiencies[det_num_a]*efficiencies[det_num_b];
        else
          det_pair_data[det_num_a][det_num_b] /=
	    efficiencies[det_num_a]*efficiencies[det_num_b];
      }
}

END_NAMESPACE_STIR
