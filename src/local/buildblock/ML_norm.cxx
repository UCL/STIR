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
#include "stir/display.h"

START_NAMESPACE_STIR

DetPairData::DetPairData()
{}

DetPairData::DetPairData(const IndexRange<2>& range)
:base_type(range), num_detectors(range.get_length())
{
}

DetPairData& 
DetPairData::operator=(const DetPairData& other)
{
  base_type::operator=(other);
  num_detectors = other.num_detectors;
  return *this;
}

float & DetPairData::operator()(const int a, const int b)
{
  return (*this)[a][b<get_min_index(a) ? b+num_detectors: b];
}

float DetPairData::operator()(const int a, const int b) const
{
  return (*this)[a][b<get_min_index(a) ? b+num_detectors: b];
}

bool 
DetPairData::
is_in_data(const int a, const int b) const
{
  if (b>=get_min_index(a))
    return b<=get_max_index(a);
  else
    return b+num_detectors<=get_max_index(a);
}

void DetPairData::fill(const float d)
{
  base_type::fill(d);
}

void DetPairData::grow(const IndexRange<2>& range)
{
  base_type::grow(range);
  num_detectors=range.get_length();
}

int DetPairData::get_min_index() const
{
  return base_type::get_min_index();
}

int DetPairData::get_max_index() const
{
  return base_type::get_max_index();
}

int DetPairData::get_min_index(const int a) const
{
  return (*this)[a].get_min_index();
}

int DetPairData::get_max_index(const int a) const
{
  return (*this)[a].get_max_index();
}

float DetPairData::sum() const
{
  return base_type::sum();
}

float DetPairData::sum(const int a) const
{
  return (*this)[a].sum();
}

float DetPairData::find_max() const
{
  return base_type::find_max();
}

float DetPairData::find_min() const
{
  return base_type::find_min();
}

int DetPairData::get_num_detectors() const
{
  return num_detectors;
}

void display(const DetPairData& det_pair_data, const char * const title)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  Array<2,float> full_data(IndexRange2D(num_detectors,num_detectors));
  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
       full_data[a%num_detectors][b%num_detectors] =
         det_pair_data(a,b);
  display(full_data,title);
}

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
  const int fan_size = 
    2*max(proj_data_info.get_max_tangential_pos_num(),
          -proj_data_info.get_min_tangential_pos_num()) + 1;
  // fan will range from -half_fan_size to +half_fan_size (i.e. an odd number of elements)
  const int half_fan_size = fan_size/2;

  IndexRange<2> fan_indices;
  fan_indices.grow(0,num_detectors-1);
  for (int a = 0; a < num_detectors; ++a)
  {
    fan_indices[a] = 
      IndexRange<1>(a+num_detectors/2-half_fan_size,
                    a+num_detectors/2+half_fan_size);
  }
  det_pair_data.grow(fan_indices);
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

	det_pair_data(det_num_a,det_num_b) =
	  (*pos_sino_ptr)[view_num][tang_pos_num];
	det_pair_data(det_num_b,det_num_a) =
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

  const int num_detectors = det_pair_data.get_num_detectors();
  assert(proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring() == num_detectors);

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
          det_pair_data(det_num_a,det_num_b);
        if (segment_num!=0)
	  (*neg_sino_ptr)[view_num][tang_pos_num] =
            det_pair_data(det_num_b,det_num_a);
      }
  proj_data.set_sinogram(*pos_sino_ptr);
  if (segment_num != 0)
    proj_data.set_sinogram(*neg_sino_ptr);
}


void apply_block_norm(DetPairData& det_pair_data, const BlockData& block_data, const bool apply)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  const int num_blocks = block_data.get_length();
  const int num_crystals_per_block = num_detectors/num_blocks;
  assert(num_blocks * num_crystals_per_block == num_detectors);
  
  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
      {
	// note: add 2*num_detectors to newb to avoid using mod with negative numbers
	if (det_pair_data(a,b) == 0)
	  continue;
        if (apply)
          det_pair_data(a,b) *=
	    block_data[a/num_crystals_per_block][(b/num_crystals_per_block)%num_blocks];
        else
          det_pair_data(a,b) /=
	    block_data[a/num_crystals_per_block][(b/num_crystals_per_block)%num_blocks];
      }
}

void apply_geo_norm(DetPairData& det_pair_data, const GeoData& geo_data, const bool apply)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  const int num_crystals_per_block = geo_data.get_length()*2;

  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
      {
	if (det_pair_data(a,b) == 0)
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
          det_pair_data(a,b) *=
	    geo_data[newa][(2*num_detectors + newb)%num_detectors];
        else
          det_pair_data(a,b) /=
	    geo_data[newa][(2*num_detectors + newb)%num_detectors];
      }
}

void apply_efficiencies(DetPairData& det_pair_data, const Array<1,float>& efficiencies, const bool apply)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
      {
	if (det_pair_data(a,b) == 0)
	  continue;
        if (apply)
	  det_pair_data(a,b) *=
	    efficiencies[a]*efficiencies[b%num_detectors];
        else
          det_pair_data(a,b) /=
	    efficiencies[a]*efficiencies[b%num_detectors];
      }
}

END_NAMESPACE_STIR
