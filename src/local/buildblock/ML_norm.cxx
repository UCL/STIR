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
#include "stir/SegmentBySinogram.h"

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


//************ 3D


FanProjData::FanProjData()
{}

FanProjData::FanProjData(const IndexRange<4>& range)
:base_type(range), num_rings(range.get_length()), num_detectors(range[range.get_min_index()].get_length())
{
}

FanProjData::
FanProjData(const int num_rings, const int num_detectors, const int max_delta, const int fan_size)
{
  // fan will range from -half_fan_size to +half_fan_size (i.e. an odd number of elements)
  const int half_fan_size = fan_size/2;

  IndexRange<4> fan_indices;
  fan_indices.grow(0,num_rings-1);
  for (int ra = 0; ra < num_rings; ++ra)
  {
    const int min_rb = max(ra-max_delta, 0);
    const int max_rb = min(ra+max_delta, num_rings-1);
    fan_indices[ra].grow(0,num_detectors-1);
    for (int a = 0; a < num_detectors; ++a)
    {
      fan_indices[ra][a].grow(min_rb, max_rb);
      for (int rb = min_rb; rb <= max_rb; ++rb)
      fan_indices[ra][a][rb] = 
        IndexRange<1>(a+num_detectors/2-half_fan_size,
                      a+num_detectors/2+half_fan_size);
    }
  }
  grow(fan_indices);
  fill(0);
}

FanProjData& 
FanProjData::operator=(const FanProjData& other)
{
  base_type::operator=(other);
  num_detectors = other.num_detectors;
  num_rings = other.num_rings;
  return *this;
}

float & FanProjData::operator()(const int ra, const int a, const int rb, const int b)
{
  return (*this)[ra][a][rb][b<get_min_b(a) ? b+num_detectors: b];
}

float FanProjData::operator()(const int ra, const int a, const int rb, const int b) const
{
  return (*this)[ra][a][rb][b<get_min_b(a) ? b+num_detectors: b];
}

bool 
FanProjData::
is_in_data(const int ra, const int a, const int rb, const int b) const
{
  if (rb<(*this)[ra][a].get_min_index() || rb >(*this)[ra][a].get_max_index())
    return false;
  if (b>=get_min_b(a))
    return b<=get_max_b(a);
  else
    return b+num_detectors<=get_max_b(a);
}

void FanProjData::fill(const float d)
{
  base_type::fill(d);
}

void FanProjData::grow(const IndexRange<4>& range)
{
  base_type::grow(range);
  num_rings =range.get_length();
  num_detectors = range[range.get_min_index()].get_length();
}

int FanProjData::get_min_index() const
{
  return base_type::get_min_index();
}

int FanProjData::get_max_index() const
{
  return base_type::get_max_index();
}


int FanProjData::get_min_a() const
{
  return (*this)[get_min_index()].get_min_index();
}

int FanProjData::get_max_a() const
{
  return (*this)[get_min_index()].get_max_index();
}


int FanProjData::get_min_rb(const int ra) const
{
  return (*this)[ra][(*this)[ra].get_min_index()].get_min_index();
}

int FanProjData::get_max_rb(const int ra) const
{
  return (*this)[ra][(*this)[ra].get_min_index()].get_max_index();
}

int FanProjData::get_min_b(const int a) const
{
  return (*this)[get_min_index()][a][(*this)[get_min_index()][a].get_min_index()].get_min_index();
}

int FanProjData::get_max_b(const int a) const
{
  return (*this)[get_min_index()][a][(*this)[get_min_index()][a].get_min_index()].get_max_index();
}


float FanProjData::sum() const
{
  return base_type::sum();
}

float FanProjData::sum(const int ra, const int a) const
{
  return (*this)[ra][a].sum();
}

float FanProjData::find_max() const
{
  return base_type::find_max();
}

float FanProjData::find_min() const
{
  return base_type::find_min();
}

int FanProjData::get_num_detectors() const
{
  return num_detectors;
}


int FanProjData::get_num_rings() const
{
  return num_rings;
}

void display(const FanProjData& fan_data, const char * const title)
{
  const int num_rings = fan_data.get_num_rings();
  const int num_detectors = fan_data.get_num_detectors();
  Array<3,float> full_data(IndexRange3D(num_rings,num_detectors,num_detectors));
  for (int ra=fan_data.get_min_index(); ra <= fan_data.get_max_index(); ++ra)
  {
    full_data.fill(0);
    for (int a = 0; a<num_detectors; ++a)
      for (int rb=fan_data.get_min_rb(ra); rb <= fan_data.get_max_rb(ra); ++rb)
        for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)      
          full_data[rb][a%num_detectors][b%num_detectors] =
             fan_data(ra,a,rb,b);
      display(full_data, full_data.find_max(), title);
  }
}

void make_fan_data(FanProjData& fan_data,
	           const ProjData& proj_data)
{
  const ProjDataInfo* proj_data_info_ptr =
    proj_data.get_proj_data_info_ptr();
  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr);

  const int num_rings = 
    proj_data_info.get_scanner_ptr()->get_num_rings();
  const int num_detectors = 
    proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  const int fan_size = 
    2*max(proj_data_info.get_max_tangential_pos_num(),
          -proj_data_info.get_min_tangential_pos_num()) + 1;
  const int max_delta = proj_data_info_ptr->get_max_segment_num();

  fan_data = FanProjData(num_rings, num_detectors, max_delta, fan_size);

  shared_ptr<SegmentBySinogram<float> > segment_ptr;      
  Bin bin;

  for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
  {
    segment_ptr = new SegmentBySinogram<float>(proj_data.get_segment_by_sinogram(bin.segment_num()));
    
    for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
	 bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
	 ++bin.axial_pos_num())
       for (bin.view_num() = 0; bin.view_num() < num_detectors/2; bin.view_num()++)
          for (bin.tangential_pos_num() = proj_data.get_min_tangential_pos_num();
	       bin.tangential_pos_num() <= proj_data.get_max_tangential_pos_num();
               ++bin.tangential_pos_num())
          {
            int ra = 0, a = 0;
            int rb = 0, b = 0;
            
            proj_data_info.get_det_pair_for_bin(a, ra, b, rb, bin);
            
            fan_data(ra, a, rb, b) =
	      fan_data(rb, b, ra, a) =
              (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()];
          }
  }
}

void set_fan_data(ProjData& proj_data,
                       const FanProjData& fan_data)
{
  const ProjDataInfo* proj_data_info_ptr =
    proj_data.get_proj_data_info_ptr();
  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr);

  const int num_rings = fan_data.get_num_rings();
  assert(num_rings == proj_data_info.get_scanner_ptr()->get_num_rings());
  const int num_detectors = fan_data.get_num_detectors();
  assert(proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring() == num_detectors);

    
  Bin bin;
  shared_ptr<SegmentBySinogram<float> > segment_ptr;    
 
  for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
  {
    segment_ptr = new SegmentBySinogram<float>(proj_data.get_empty_segment_by_sinogram(bin.segment_num()));
    
    for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
	 bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
	 ++bin.axial_pos_num())
       for (bin.view_num() = 0; bin.view_num() < num_detectors/2; bin.view_num()++)
          for (bin.tangential_pos_num() = proj_data.get_min_tangential_pos_num();
	       bin.tangential_pos_num() <= proj_data.get_max_tangential_pos_num();
               ++bin.tangential_pos_num())
          {
            int ra = 0, a = 0;
            int rb = 0, b = 0;
            
            proj_data_info.get_det_pair_for_bin(a, ra, b, rb, bin);
            
            (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()] =
              fan_data(ra, a, rb, b);
          }
    proj_data.set_segment(*segment_ptr);
  }
}

#if 0

void apply_block_norm(FanProjData& fan_data, const BlockData& block_data, const bool apply)
{
  const int num_detectors = fan_data.get_num_detectors();
  const int num_blocks = block_data.get_length();
  const int num_crystals_per_block = num_detectors/num_blocks;
  assert(num_blocks * num_crystals_per_block == num_detectors);
  
  for (int a = fan_data.get_min_index(); a <= fan_data.get_max_index(); ++a)
    for (int b = fan_data.get_min_index(a); b <= fan_data.get_max_index(a); ++b)      
      {
	// note: add 2*num_detectors to newb to avoid using mod with negative numbers
	if (fan_data(a,b) == 0)
	  continue;
        if (apply)
          fan_data(a,b) *=
	    block_data[a/num_crystals_per_block][(b/num_crystals_per_block)%num_blocks];
        else
          fan_data(a,b) /=
	    block_data[a/num_crystals_per_block][(b/num_crystals_per_block)%num_blocks];
      }
}

void apply_geo_norm(FanProjData& fan_data, const GeoData& geo_data, const bool apply)
{
  const int num_detectors = fan_data.get_num_detectors();
  const int num_crystals_per_block = geo_data.get_length()*2;

  for (int a = fan_data.get_min_index(); a <= fan_data.get_max_index(); ++a)
    for (int b = fan_data.get_min_index(a); b <= fan_data.get_max_index(a); ++b)      
      {
	if (fan_data(a,b) == 0)
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
          fan_data(a,b) *=
	    geo_data[newa][(2*num_detectors + newb)%num_detectors];
        else
          fan_data(a,b) /=
	    geo_data[newa][(2*num_detectors + newb)%num_detectors];
      }
}
#endif

void apply_efficiencies(FanProjData& fan_data, const DetectorEfficiencies& efficiencies, const bool apply)
{
  const int num_detectors = fan_data.get_num_detectors();
  for (int ra = fan_data.get_min_index(); ra <= fan_data.get_max_index(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      for (int rb = fan_data.get_min_rb(ra); rb <= fan_data.get_max_rb(ra); ++rb)
        for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)      
        {
	  if (fan_data(ra,a,rb,b) == 0)
	    continue;
          if (apply)
	    fan_data(ra,a,rb,b) *=
	      efficiencies[ra][a]*efficiencies[rb][b%num_detectors];
          else
            fan_data(ra,a,rb,b) /=
	      efficiencies[ra][a]*efficiencies[rb][b%num_detectors];
      }
}


END_NAMESPACE_STIR
