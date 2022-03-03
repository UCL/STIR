/*
    Copyright (C) 2001- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2016, 2020, 2021, University College London
    Copyright (C) 2016-2017, PETsys Electronics
    Copyright (C) 2021, Gefei Chen
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup buildblock

  \brief utilities for finding normalisation factors using an ML approach

  \author Kris Thielemans
  \author Tahereh Niknejad
  \author Gefei Chen

*/

#include "stir/ML_norm.h"
#include "stir/SegmentBySinogram.h"
#include "stir/stream.h"

#include <algorithm>
using std::min;
using std::max;

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

void make_det_pair_data(DetPairData& det_pair_data,
			const ProjDataInfo& proj_data_info_general_type,
			const int segment_num,
			const int ax_pos_num)
{
  if(proj_data_info_general_type.get_scanner_ptr()->get_scanner_geometry()== "BlocksOnCylindrical")
    {
      const ProjDataInfoBlocksOnCylindricalNoArcCorr& proj_data_info =
        dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr&>(proj_data_info_general_type);

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
    }
  else
    {
      const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
	dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(proj_data_info_general_type);

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
    }
}

void make_det_pair_data(DetPairData& det_pair_data,
			const ProjData& proj_data,
			const int segment_num,
			const int ax_pos_num)
{
  if(proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry()== "BlocksOnCylindrical")
    {
     make_det_pair_data(det_pair_data,
                         *proj_data.get_proj_data_info_sptr(),
                         segment_num,
                         ax_pos_num);
      const int num_detectors =
        det_pair_data.get_num_detectors();
      const ProjDataInfoBlocksOnCylindricalNoArcCorr& proj_data_info =
        dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr&>(*proj_data.get_proj_data_info_sptr());

      shared_ptr<Sinogram<float> >
        pos_sino_ptr(new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,segment_num)));
      shared_ptr<Sinogram<float> > neg_sino_ptr;
      if (segment_num == 0)
        neg_sino_ptr = pos_sino_ptr;
      else
        neg_sino_ptr.
          reset(new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,-segment_num)));


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
  else
    {
      make_det_pair_data(det_pair_data,
			 *proj_data.get_proj_data_info_sptr(),
			 segment_num,
			 ax_pos_num);
      const int num_detectors = 
	det_pair_data.get_num_detectors();
      const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
	dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data.get_proj_data_info_sptr());

      shared_ptr<Sinogram<float> > 
	pos_sino_ptr(new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,segment_num)));
      shared_ptr<Sinogram<float> > neg_sino_ptr;
      if (segment_num == 0)
	neg_sino_ptr = pos_sino_ptr;
      else
	neg_sino_ptr.
	  reset(new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,-segment_num)));
  
    
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
}

void set_det_pair_data(ProjData& proj_data,
                       const DetPairData& det_pair_data,
			const int segment_num,
			const int ax_pos_num)
{
  const shared_ptr<const ProjDataInfo> proj_data_info_sptr =
    proj_data.get_proj_data_info_sptr();
  if(proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry()== "BlocksOnCylindrical")
    {
      const ProjDataInfoBlocksOnCylindricalNoArcCorr& proj_data_info =
        dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr&>(*proj_data_info_sptr);

      const int num_detectors = det_pair_data.get_num_detectors();
      assert(proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring() == num_detectors);

      shared_ptr<Sinogram<float> >
        pos_sino_ptr(new Sinogram<float>(proj_data.get_empty_sinogram(ax_pos_num,segment_num)));
      shared_ptr<Sinogram<float> > neg_sino_ptr;
      if (segment_num != 0)
        neg_sino_ptr.
          reset(new Sinogram<float>(proj_data.get_empty_sinogram(ax_pos_num,-segment_num)));


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
  else
    {
      const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
	dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_sptr);

      const int num_detectors = det_pair_data.get_num_detectors();
      assert(proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring() == num_detectors);

      shared_ptr<Sinogram<float> > 
	pos_sino_ptr(new Sinogram<float>(proj_data.get_empty_sinogram(ax_pos_num,segment_num)));
      shared_ptr<Sinogram<float> > neg_sino_ptr;
      if (segment_num != 0)
	neg_sino_ptr.
	  reset(new Sinogram<float>(proj_data.get_empty_sinogram(ax_pos_num,-segment_num)));
  
    
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


void make_fan_sum_data(Array<1,float>& data_fan_sums, const DetPairData& det_pair_data)
{
  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    data_fan_sums[a] = det_pair_data.sum(a);
}

void make_geo_data(GeoData& geo_data, const DetPairData& det_pair_data)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  const int num_crystals_per_block = geo_data.get_length()*2;
  const int num_blocks = num_detectors / num_crystals_per_block;
  assert(num_blocks * num_crystals_per_block == num_detectors);

  // TODO optimise
  DetPairData work = det_pair_data;
  work.fill(0);

  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
        {
	// mirror symmetry
	work(a,b) = 
	  det_pair_data(a,b) + 
	  det_pair_data(num_detectors-1-a,(2*num_detectors-1-b)%num_detectors);
      }

  geo_data.fill(0);

  for (int crystal_num_a = 0; crystal_num_a < num_crystals_per_block/2; ++crystal_num_a)
    for (int det_num_b = det_pair_data.get_min_index(crystal_num_a); det_num_b <= det_pair_data.get_max_index(crystal_num_a); ++det_num_b)      
      {
	for (int block_num = 0; block_num<num_blocks; ++block_num)
	  {
	    const int det_inc = block_num * num_crystals_per_block;
            const int new_det_num_a = (crystal_num_a+det_inc)%num_detectors;
            const int new_det_num_b = (det_num_b+det_inc)%num_detectors;
            if (det_pair_data.is_in_data(new_det_num_a,new_det_num_b))
	      geo_data[crystal_num_a][det_num_b%num_detectors] +=
	        work(new_det_num_a,new_det_num_b);
	  }
      }
  geo_data /= 2*num_blocks;
  // TODO why normalisation?
}
 
void make_block_data(BlockData& block_data, const DetPairData& det_pair_data)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  const int num_blocks = block_data.get_length();
  const int num_crystals_per_block = num_detectors/num_blocks;
  assert(num_blocks * num_crystals_per_block == num_detectors);

  block_data.fill(0);
  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
      {
	block_data[a/num_crystals_per_block][(b/num_crystals_per_block)%num_blocks] += 
	  det_pair_data(a,b);
      }
  
  block_data /= square(num_crystals_per_block);
  // TODO why normalisation?
}
 
void iterate_efficiencies(Array<1,float>& efficiencies,
			  const Array<1,float>& data_fan_sums,
			  const DetPairData& model)
{
  const int num_detectors = efficiencies.get_length();

  for (int a = 0; a < num_detectors; ++a)
    {
      if (data_fan_sums[a] == 0)
	efficiencies[a] = 0;
      else
	{
          //const float denominator = inner_product(efficiencies,model[a]);
	  float denominator = 0;
           for (int b = model.get_min_index(a); b <= model.get_max_index(a); ++b)      
  	    denominator += efficiencies[b%num_detectors]*model(a,b);
	  efficiencies[a] = data_fan_sums[a] / denominator;
	}
    }
}

void iterate_geo_norm(GeoData& norm_geo_data,
		      const GeoData& measured_geo_data,
		      const DetPairData& model)
{
  make_geo_data(norm_geo_data, model);
  //norm_geo_data = measured_geo_data / norm_geo_data;
  const int num_detectors = model.get_num_detectors();
  const int num_crystals_per_block = measured_geo_data.get_length()*2;
  const float threshold = measured_geo_data.find_max()/10000.F;
  for (int a = 0; a < num_crystals_per_block/2; ++a)
    for (int b = 0; b < num_detectors; ++b)      
      {
	norm_geo_data[a][b] =
	  (measured_geo_data[a][b]>=threshold ||
	   measured_geo_data[a][b] < 10000*norm_geo_data[a][b])
	  ? measured_geo_data[a][b] / norm_geo_data[a][b]
	  : 0;
      }
}
  
void iterate_block_norm(BlockData& norm_block_data,
		      const BlockData& measured_block_data,
		      const DetPairData& model)
{
  make_block_data(norm_block_data, model);
  //norm_block_data = measured_block_data / norm_block_data;
  const int num_blocks = norm_block_data.get_length();
  const float threshold = measured_block_data.find_max()/10000.F;
  for (int a = 0; a < num_blocks; ++a)
    for (int b = 0; b < num_blocks; ++b)      
      {
	norm_block_data[a][b] =
	  (measured_block_data[a][b]>=threshold ||
	   measured_block_data[a][b] < 10000*norm_block_data[a][b])
	  ? measured_block_data[a][b] / norm_block_data[a][b]
	  : 0;
      }
}

double KL(const DetPairData& d1, const DetPairData& d2, const double threshold)
{
  double sum=0;
  for (int a = d1.get_min_index(); a <= d1.get_max_index(); ++a)
    {
      double bsum=0;
      for (int b = d1.get_min_index(a); b <= d1.get_max_index(a); ++b)      
        bsum += KL(d1(a,b), d2(a,b), threshold);
      sum += bsum;
    }
  return static_cast<double>(sum);
}


//////////   *******   GeoData3D   ********  //////
GeoData3D::GeoData3D()
{}

GeoData3D::~GeoData3D()
{}


GeoData3D::
GeoData3D(const int num_axial_crystals_per_block, const int half_num_transaxial_crystals_per_block, const int num_rings, const int num_detectors_per_ring)
:num_axial_crystals_per_block(num_axial_crystals_per_block), half_num_transaxial_crystals_per_block(half_num_transaxial_crystals_per_block),
num_rings(num_rings), num_detectors_per_ring(num_detectors_per_ring)
{
    
   // assert(max_ring_diff<num_axial_crystals_per_block-1);
    // I assumed that max_ring_diff is always >= (num_axial_crystals_per_block -1)

    
    IndexRange<4> fan_indices;
    fan_indices.grow(0,num_axial_crystals_per_block-1);
    for (int ra = 0; ra < num_axial_crystals_per_block; ++ra)
    {
        const int min_rb = ra;
        const int max_rb = num_rings-1; // I assumed max ring diff is num_ring_1
        fan_indices[ra].grow(0,half_num_transaxial_crystals_per_block-1);
        for (int a = 0; a < half_num_transaxial_crystals_per_block; ++a)
        {
            // store only 1 half of data as ra,a,rb,b = rb,b,ra,a
            fan_indices[ra][a].grow(min_rb, max_rb);
            for (int rb = min_rb; rb <= max_rb; ++rb)
                fan_indices[ra][a][rb] =
                IndexRange<1>(a,
                              a+num_detectors_per_ring -1); // I assumed fan size is number of detector per ring  - 2
        }
    }
    grow(fan_indices);
    fill(0);
}

GeoData3D&
GeoData3D::operator=(const GeoData3D& other)
{
    base_type::operator=(other);
    num_axial_crystals_per_block = other.num_axial_crystals_per_block;
    half_num_transaxial_crystals_per_block = other.half_num_transaxial_crystals_per_block;
    num_rings = other.num_rings;
    num_detectors_per_ring = other.num_detectors_per_ring;
    return *this;
}

/*float & GeoData3D::operator()(const int ra, const int a, const int rb, const int b)
{
    assert(a>=0);
    assert(b>=0);
    return
    ra<rb
    ? (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b]
    : b < half_num_transaxial_crystals_per_block ? (*this)[rb][b%num_detectors_per_ring][ra][a<get_min_b(b%num_detectors_per_ring) ? a+num_detectors_per_ring: a] : (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b];
}
float GeoData3D::operator()(const int ra, const int a, const int rb, const int b) const
{
   assert(a>=0);
    assert(b>=0);
    return
    ra<rb
    ? (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b]
    : b < half_num_transaxial_crystals_per_block  ? (*this)[rb][b%num_detectors_per_ring][ra][a<get_min_b(b%num_detectors_per_ring) ? a+num_detectors_per_ring: a] : (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b];
} */

float & GeoData3D::operator()(const int ra, const int a, const int rb, const int b)
{
    assert(a>=0);
    assert(b>=0);
    return

    (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b];
   
}

float GeoData3D::operator()(const int ra, const int a, const int rb, const int b) const
{
   assert(a>=0);
    assert(b>=0);
    return

    (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b];

} 

bool
GeoData3D::
is_in_data(const int ra, const int a, const int rb, const int b) const
{
    assert(a>=0);
    assert(b>=0);
    if (rb<(*this)[ra][a].get_min_index() || rb >(*this)[ra][a].get_max_index())
        return false;
    if (b>=get_min_b(a))
        return b<=get_max_b(a);
    else
        return b+num_detectors_per_ring<=get_max_b(a);
}
void GeoData3D::fill(const float d)
{
    base_type::fill(d);
}


int GeoData3D::get_min_ra() const
{
    return base_type::get_min_index();
}

int GeoData3D::get_max_ra() const
{
    return base_type::get_max_index();
}


int GeoData3D::get_min_a() const
{
    return (*this)[get_min_index()].get_min_index();
}

int GeoData3D::get_max_a() const
{
    return (*this)[get_min_index()].get_max_index();
}


int GeoData3D::get_min_rb(const int ra) const
{
    return 0;
    // next is no longer true because we store only half the data
    //return (*this)[ra][(*this)[ra].get_min_index()].get_min_index();
}


int GeoData3D::get_max_rb(const int ra) const
{
    return (*this)[ra][(*this)[ra].get_min_index()].get_max_index();
}

int GeoData3D::get_min_b(const int a) const
{
    return (*this)[get_min_index()][a][(*this)[get_min_index()][a].get_min_index()].get_min_index();
}

int GeoData3D::get_max_b(const int a) const
{
    return (*this)[get_min_index()][a][(*this)[get_min_index()][a].get_min_index()].get_max_index();
}


float GeoData3D::sum() const
{
    //return base_type::sum();
    float sum = 0;
    for (int ra=get_min_ra(); ra <= get_max_ra(); ++ra)
        for (int a = get_min_a(); a <= get_max_a(); ++a)
            sum += this->sum(ra,a);
    return sum;
}

float GeoData3D::sum(const int ra, const int a) const
{
    //return (*this)[ra][a].sum();
    float sum = 0;
    for (int rb=get_min_rb(ra); rb <= get_max_rb(ra); ++rb)
        for (int b = get_min_b(a); b <= get_max_b(a); ++b)
            sum += (*this)(ra,a,rb,b%num_detectors_per_ring);
    return sum;
}

float GeoData3D::find_max() const
{
    return base_type::find_max();
}

float GeoData3D::find_min() const
{
    return base_type::find_min();
}

int GeoData3D::get_num_axial_crystals_per_block() const
{
    return num_axial_crystals_per_block;
}

int GeoData3D::get_half_num_transaxial_crystals_per_block() const
{
    return half_num_transaxial_crystals_per_block;
}


std::ostream& operator<<(std::ostream& s, const GeoData3D& geo_data)
{
    return s << static_cast<GeoData3D::base_type>(geo_data);
}

std::istream& operator>>(std::istream& s, GeoData3D& geo_data)
{
    s >> static_cast<GeoData3D::base_type&>(geo_data);
    if (!s)
        return s;
    geo_data.half_num_transaxial_crystals_per_block = geo_data.get_max_a() - geo_data.get_min_a() + 1;
    geo_data.num_axial_crystals_per_block = geo_data.get_max_ra() - geo_data.get_min_ra() + 1;
    
    
    const int max_delta = geo_data[0][0].get_length()-1;
    const int num_detectors_per_ring =
    geo_data[0][0][0].get_length();
    geo_data.num_detectors_per_ring = num_detectors_per_ring;
    geo_data.num_rings = max_delta+1;
    
    

    
    for (int ra = 0; ra < geo_data.num_axial_crystals_per_block; ++ra)
        

    {
        const int min_rb = ra;
        const int max_rb = geo_data.num_rings-1;
        for (int a = 0; a < geo_data.half_num_transaxial_crystals_per_block; ++a)
        {

            if (geo_data[ra][a].get_length() != max_rb - max(ra,min_rb) + 1)
            {
                warning("Reading GeoData3D: inconsistent length %d for rb at ra=%d, a=%d, "
                        "Expected length %d\n",
                        geo_data[ra][a].get_length(), ra, a, max_rb - max(ra,min_rb) + 1);
            }
            geo_data[ra][a].set_offset(max(ra,min_rb));
            for (int rb = geo_data[ra][a].get_min_index(); rb <= geo_data[ra][a].get_max_index(); ++rb)
            {
                if (geo_data[ra][a][rb].get_length() != num_detectors_per_ring)
                {
                    warning("Reading GeoData3D: inconsistent length %d for b at ra=%d, a=%d, rb=%d\n"
                            "Expected length %d\n", 
                            geo_data[ra][a][rb].get_length(), ra, a, rb, num_detectors_per_ring);
                }
                geo_data[ra][a][rb].set_offset(a);
            }
        }
    }
    
    return s;
}


////////////////////////////////////////////////////




FanProjData::FanProjData()
{}

FanProjData::~FanProjData()
{}

#if 0
FanProjData::FanProjData(const IndexRange<4>& range)
:base_type(range), num_rings(range.get_length()), num_detectors_per_ring(range[range.get_min_index()].get_length())
{
}
#endif

FanProjData::
FanProjData(const int num_rings, const int num_detectors_per_ring, const int max_ring_diff, const int fan_size)
: num_rings(num_rings), num_detectors_per_ring(num_detectors_per_ring),
  max_ring_diff(max_ring_diff), half_fan_size(fan_size/2)
{
  assert(num_detectors_per_ring%2 == 0);
  assert(max_ring_diff<num_rings);
  assert(fan_size < num_detectors_per_ring);
  
  IndexRange<4> fan_indices;
  fan_indices.grow(0,num_rings-1);
  for (int ra = 0; ra < num_rings; ++ra)
  {
    const int min_rb = max(ra-max_ring_diff, 0);
    const int max_rb = min(ra+max_ring_diff, num_rings-1);
    fan_indices[ra].grow(0,num_detectors_per_ring-1);
    for (int a = 0; a < num_detectors_per_ring; ++a)
    {
      // store only 1 half of data as ra,a,rb,b = rb,b,ra,a
      fan_indices[ra][a].grow(max(ra,min_rb), max_rb);
      for (int rb = max(ra,min_rb); rb <= max_rb; ++rb)
      fan_indices[ra][a][rb] = 
        IndexRange<1>(a+num_detectors_per_ring/2-half_fan_size,
                      a+num_detectors_per_ring/2+half_fan_size);
    }
  }
  grow(fan_indices);
  fill(0);
}

FanProjData& 
FanProjData::operator=(const FanProjData& other)
{
  base_type::operator=(other);
  num_detectors_per_ring = other.num_detectors_per_ring;
  num_rings = other.num_rings;
  max_ring_diff = other.max_ring_diff;
  half_fan_size = other.half_fan_size;
  return *this;
}

float & FanProjData::operator()(const int ra, const int a, const int rb, const int b)
{
  assert(a>=0);
  assert(b>=0);
  return 
    ra<rb 
    ? (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b]
    : (*this)[rb][b%num_detectors_per_ring][ra][a<get_min_b(b%num_detectors_per_ring) ? a+num_detectors_per_ring: a];
}

float FanProjData::operator()(const int ra, const int a, const int rb, const int b) const
{  
  assert(a>=0);
  assert(b>=0);
  return 
    ra<rb 
    ? (*this)[ra][a%num_detectors_per_ring][rb][b<get_min_b(a) ? b+num_detectors_per_ring: b]
    : (*this)[rb][b%num_detectors_per_ring][ra][a<get_min_b(b%num_detectors_per_ring) ? a+num_detectors_per_ring: a];
}

bool 
FanProjData::
is_in_data(const int ra, const int a, const int rb, const int b) const
{
  assert(a>=0);
  assert(b>=0);
  if (rb<(*this)[ra][a].get_min_index() || rb >(*this)[ra][a].get_max_index())
    return false;
  if (b>=get_min_b(a))
    return b<=get_max_b(a);
  else
    return b+num_detectors_per_ring<=get_max_b(a);
}

void FanProjData::fill(const float d)
{
  base_type::fill(d);
}

#if 0
void FanProjData::grow(const IndexRange<4>& range)
{
  base_type::grow(range);
  num_rings =range.get_length();
  num_detectors_per_ring = range[range.get_min_index()].get_length();
}
#endif

int FanProjData::get_min_ra() const
{
  return base_type::get_min_index();
}

int FanProjData::get_max_ra() const
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
  return max(ra-max_ring_diff, 0); 
  // next is no longer true because we store only half the data
  //return (*this)[ra][(*this)[ra].get_min_index()].get_min_index();
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
  //return base_type::sum();
  float sum = 0;
  for (int ra=get_min_ra(); ra <= get_max_ra(); ++ra)
    for (int a = get_min_a(); a <= get_max_a(); ++a)      
      sum += this->sum(ra,a);
  return sum;
}

float FanProjData::sum(const int ra, const int a) const
{
  //return (*this)[ra][a].sum();
  float sum = 0;
  for (int rb=get_min_rb(ra); rb <= get_max_rb(ra); ++rb)
    for (int b = get_min_b(a); b <= get_max_b(a); ++b)      
      sum += (*this)(ra,a,rb,b%num_detectors_per_ring);
  return sum;
}

float FanProjData::find_max() const
{
  return base_type::find_max();
}

float FanProjData::find_min() const
{
  return base_type::find_min();
}

int FanProjData::get_num_detectors_per_ring() const
{
  return num_detectors_per_ring;
}


int FanProjData::get_num_rings() const
{
  return num_rings;
}

std::ostream& operator<<(std::ostream& s, const FanProjData& fan_data)
{
  return s << static_cast<FanProjData::base_type>(fan_data);
}

std::istream& operator>>(std::istream& s, FanProjData& fan_data)
{
  s >> static_cast<FanProjData::base_type&>(fan_data);
  if (!s)
    return s;
  fan_data.num_detectors_per_ring = fan_data.get_max_a() - fan_data.get_min_a() + 1;
  fan_data.num_rings = fan_data.get_max_ra() - fan_data.get_min_ra() + 1;
      
  //int max_delta = 0;
  //for (int ra = 0; ra < fan_data.num_rings; ++ra)
  //  max_delta = max(max_delta,fan_data[ra][0].get_length()-1);
  const int max_delta = fan_data[0][0].get_length()-1;
  const int half_fan_size = 
    fan_data[0][0][0].get_length()/2;
  fan_data.half_fan_size = half_fan_size;
  fan_data.max_ring_diff = max_delta;

  for (int ra = 0; ra < fan_data.num_rings; ++ra)
  {
    const int min_rb = max(ra-max_delta, 0);
    const int max_rb = min(ra+max_delta, fan_data.num_rings-1);
    for (int a = 0; a < fan_data.num_detectors_per_ring; ++a)
    {
      if (fan_data[ra][a].get_length() != max_rb - max(ra,min_rb) + 1)
      {
        warning("Reading FanProjData: inconsistent length %d for rb at ra=%d, a=%d, "
                "Expected length %d\n", 
                fan_data[ra][a].get_length(), ra, a, max_rb - max(ra,min_rb) + 1);
      }
      fan_data[ra][a].set_offset(max(ra,min_rb));
      for (int rb = fan_data[ra][a].get_min_index(); rb <= fan_data[ra][a].get_max_index(); ++rb)
      {
        if (fan_data[ra][a][rb].get_length() != 2*half_fan_size+1)
        {
          warning("Reading FanProjData: inconsistent length %d for b at ra=%d, a=%d, rb=%d\n"
                 "Expected length %d\n", 
                  fan_data[ra][a][rb].get_length(), ra, a, rb, 2*half_fan_size+1);
        }
        fan_data[ra][a][rb].set_offset(a+fan_data.num_detectors_per_ring/2-half_fan_size);
      }
    }
  }

  return s;
}

shared_ptr<const ProjDataInfoCylindricalNoArcCorr>
get_fan_info(int& num_rings, int& num_detectors_per_ring, 
	     int& max_ring_diff, int& fan_size, 
	     const ProjDataInfo& proj_data_info)
{
  const ProjDataInfoCylindricalNoArcCorr * const proj_data_info_ptr = 
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>(&proj_data_info);
  if (proj_data_info_ptr == 0)
  {
    error("Can only process not arc-corrected data\n");
  }
  if (proj_data_info_ptr->get_view_mashing_factor()>1)
  {
    error("Can only process data without mashing of views\n");
  }
  if (proj_data_info_ptr->get_max_ring_difference(0)>0)
  {
    error("Can only process data without axial compression (i.e. span=1)\n");
  }
  num_rings = 
    proj_data_info.get_scanner_ptr()->get_num_rings();
  num_detectors_per_ring = 
    proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  const int half_fan_size = 
    min(proj_data_info.get_max_tangential_pos_num(),
          -proj_data_info.get_min_tangential_pos_num());
  fan_size = 2*half_fan_size+1;
  max_ring_diff = proj_data_info_ptr->get_max_segment_num();

  shared_ptr<ProjDataInfoCylindricalNoArcCorr> 
    ret_value(dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(proj_data_info_ptr->clone()));
  return ret_value;
	    
}


shared_ptr<const ProjDataInfoBlocksOnCylindricalNoArcCorr>
get_fan_info_block(int& num_rings, int& num_detectors_per_ring,
             int& max_ring_diff, int& fan_size,
             const ProjDataInfo& proj_data_info)
{
  const ProjDataInfoBlocksOnCylindricalNoArcCorr * const proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr * const>(&proj_data_info);
  if (proj_data_info_ptr == 0)
  {
    error("Can only process not arc-corrected data\n");
  }
  if (proj_data_info_ptr->get_view_mashing_factor()>1)
  {
    error("Can only process data without mashing of views\n");
  }
  if (proj_data_info_ptr->get_max_ring_difference(0)>0)
  {
    error("Can only process data without axial compression (i.e. span=1)\n");
  }
  num_rings =
    proj_data_info.get_scanner_ptr()->get_num_rings();
  num_detectors_per_ring =
    proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  const int half_fan_size =
    min(proj_data_info.get_max_tangential_pos_num(),
          -proj_data_info.get_min_tangential_pos_num());
  fan_size = 2*half_fan_size+1;
  max_ring_diff = proj_data_info_ptr->get_max_segment_num();

  shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr>
    ret_value(dynamic_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr *>(proj_data_info_ptr->clone()));
  return ret_value;

}


/// **** This function make fan_data from projecion file while removing the intermodule gaps **** ////
/// *** fan_data doesn't have gaps, proj_data has gaps *** ///
void make_fan_data_remove_gaps(FanProjData& fan_data,
                   const ProjData& proj_data)
{
    int num_rings;
    int num_detectors_per_ring;
    int fan_size;
    int max_delta;
    if(proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry()== "BlocksOnCylindrical")
      {
        shared_ptr<const ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_ptr =
          get_fan_info_block(num_rings, num_detectors_per_ring, max_delta, fan_size,
                       *proj_data.get_proj_data_info_sptr());

        const int half_fan_size = fan_size/2;


        const int num_virtual_axial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_virtual_axial_crystals_per_block();

        const int num_virtual_transaxial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_virtual_transaxial_crystals_per_block();

        const int num_transaxial_blocks =
          proj_data_info_ptr ->get_scanner_sptr()->
          get_num_transaxial_blocks();
        const int num_axial_blocks =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_axial_blocks();
        const int num_transaxial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_transaxial_crystals_per_block();
        const int num_axial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_axial_crystals_per_block();

        const int num_physical_transaxial_crystals_per_block = num_transaxial_crystals_per_block - num_virtual_transaxial_crystals_per_block;

        const int num_physical_axial_crystals_per_block = num_axial_crystals_per_block - num_virtual_axial_crystals_per_block;


        const int num_transaxial_blocks_in_fansize = fan_size/(num_transaxial_crystals_per_block);
        const int new_fan_size = fan_size - num_transaxial_blocks_in_fansize*num_virtual_transaxial_crystals_per_block;
        const int new_half_fan_size = new_fan_size/2;
        const int num_axial_blocks_in_max_delta = max_delta/(num_axial_crystals_per_block);
        const int new_max_delta = max_delta - (num_axial_blocks_in_max_delta)*num_virtual_axial_crystals_per_block;
        const int num_physical_detectors_per_ring = num_detectors_per_ring - num_transaxial_blocks*num_virtual_transaxial_crystals_per_block;
        const int num_physical_rings = num_rings - (num_axial_blocks-1)*num_virtual_axial_crystals_per_block;

        fan_data = FanProjData(num_physical_rings, num_physical_detectors_per_ring, new_max_delta, 2*new_half_fan_size+1);


        shared_ptr<SegmentBySinogram<float> > segment_ptr;
        Bin bin;

        for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
          {
            segment_ptr.reset(new SegmentBySinogram<float>(proj_data.get_segment_by_sinogram(bin.segment_num())));

            for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
                 bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
                 ++bin.axial_pos_num())
              for (bin.view_num() = 0; bin.view_num() < num_detectors_per_ring/2; bin.view_num()++)
                for (bin.tangential_pos_num() = -half_fan_size;
                     bin.tangential_pos_num() <= half_fan_size;
                     ++bin.tangential_pos_num())
                  {
                    int ra = 0, a = 0;
                    int rb = 0, b = 0;

                    proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);
                    int a_in_block = a % num_transaxial_crystals_per_block;

                    if (a_in_block >= num_physical_transaxial_crystals_per_block)
                      continue;
                    int new_a = a - (a / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int ra_in_block = ra % num_axial_crystals_per_block;
                    if (ra_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_ra = ra - (ra / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;

                    int b_in_block = b % num_transaxial_crystals_per_block;
                    if (b_in_block >= num_physical_transaxial_crystals_per_block)
                        continue;
                    int new_b = b - (b / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int rb_in_block = rb % num_axial_crystals_per_block;
                    if (rb_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_rb = rb - (rb / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;

                    fan_data(new_ra, new_a, new_rb, new_b) =
                    fan_data(new_rb, new_b, new_ra, new_a) =
                    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()];
                  }
          }	
      }
    else
      {
	shared_ptr<const ProjDataInfoCylindricalNoArcCorr> proj_data_info_ptr =
	  get_fan_info(num_rings, num_detectors_per_ring, max_delta, fan_size,
		       *proj_data.get_proj_data_info_sptr());
    
	const int half_fan_size = fan_size/2;


	const int num_virtual_axial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_virtual_axial_crystals_per_block();

	const int num_virtual_transaxial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_virtual_transaxial_crystals_per_block();

	const int num_transaxial_blocks =
	  proj_data_info_ptr ->get_scanner_sptr()->
	  get_num_transaxial_blocks();
	const int num_axial_blocks =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_axial_blocks();
	const int num_transaxial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_transaxial_crystals_per_block();
	const int num_axial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_axial_crystals_per_block();

	const int num_physical_transaxial_crystals_per_block = num_transaxial_crystals_per_block - num_virtual_transaxial_crystals_per_block;

	const int num_physical_axial_crystals_per_block = num_axial_crystals_per_block - num_virtual_axial_crystals_per_block;


	const int num_transaxial_blocks_in_fansize = fan_size/(num_transaxial_crystals_per_block);
	const int new_fan_size = fan_size - num_transaxial_blocks_in_fansize*num_virtual_transaxial_crystals_per_block;
	const int new_half_fan_size = new_fan_size/2;
	const int num_axial_blocks_in_max_delta = max_delta/(num_axial_crystals_per_block);
	const int new_max_delta = max_delta - (num_axial_blocks_in_max_delta)*num_virtual_axial_crystals_per_block;
	const int num_physical_detectors_per_ring = num_detectors_per_ring - num_transaxial_blocks*num_virtual_transaxial_crystals_per_block;
	const int num_physical_rings = num_rings - (num_axial_blocks-1)*num_virtual_axial_crystals_per_block;

	fan_data = FanProjData(num_physical_rings, num_physical_detectors_per_ring, new_max_delta, 2*new_half_fan_size+1);


	shared_ptr<SegmentBySinogram<float> > segment_ptr;
	Bin bin;

	for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
	  {
	    segment_ptr.reset(new SegmentBySinogram<float>(proj_data.get_segment_by_sinogram(bin.segment_num())));
	    
	    for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
		 bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
		 ++bin.axial_pos_num())
	      for (bin.view_num() = 0; bin.view_num() < num_detectors_per_ring/2; bin.view_num()++)
                for (bin.tangential_pos_num() = -half_fan_size;
                     bin.tangential_pos_num() <= half_fan_size;
                     ++bin.tangential_pos_num())
		  {
                    int ra = 0, a = 0;
                    int rb = 0, b = 0;

                    proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);
                    int a_in_block = a % num_transaxial_crystals_per_block;

                    if (a_in_block >= num_physical_transaxial_crystals_per_block)
		      continue;
                    int new_a = a - (a / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int ra_in_block = ra % num_axial_crystals_per_block;
                    if (ra_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_ra = ra - (ra / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;

                    int b_in_block = b % num_transaxial_crystals_per_block;
                    if (b_in_block >= num_physical_transaxial_crystals_per_block)
                        continue;
                    int new_b = b - (b / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int rb_in_block = rb % num_axial_crystals_per_block;
                    if (rb_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_rb = rb - (rb / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;

                    fan_data(new_ra, new_a, new_rb, new_b) =
                    fan_data(new_rb, new_b, new_ra, new_a) =
                    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()];
		  }
	  }
      }
}


/// **** This function make proj_data from fan_data while adding the intermodule gaps **** ////
/// *** fan_data doesn't have gaps, proj_data has gaps *** ///
void set_fan_data_add_gaps(ProjData& proj_data,
                  const FanProjData& fan_data, const float gap_value)
{
    int num_rings;
    int num_detectors_per_ring;
    int fan_size;
    int max_delta;
    if(proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry()== "BlocksOnCylindrical")
      {
       shared_ptr<const ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_ptr =
          get_fan_info_block(num_rings, num_detectors_per_ring, max_delta, fan_size,
                       *proj_data.get_proj_data_info_sptr());

        const int half_fan_size = fan_size/2;

        const int num_virtual_axial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_virtual_axial_crystals_per_block();
        const int num_virtual_transaxial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_virtual_transaxial_crystals_per_block();
        const int num_transaxial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_transaxial_crystals_per_block();
        const int num_axial_crystals_per_block =
          proj_data_info_ptr->get_scanner_sptr()->
          get_num_axial_crystals_per_block();

        const int num_physical_transaxial_crystals_per_block = num_transaxial_crystals_per_block - num_virtual_transaxial_crystals_per_block;

        const int num_physical_axial_crystals_per_block = num_axial_crystals_per_block - num_virtual_axial_crystals_per_block;

        Bin bin;
        shared_ptr<SegmentBySinogram<float> > segment_ptr;

        for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
          {
            segment_ptr.reset(new SegmentBySinogram<float>(proj_data.get_empty_segment_by_sinogram(bin.segment_num())));

            for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
                 bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
                 ++bin.axial_pos_num())
              for (bin.view_num() = 0; bin.view_num() < num_detectors_per_ring/2; bin.view_num()++)
                for (bin.tangential_pos_num() = -half_fan_size;
                     bin.tangential_pos_num() <= half_fan_size;
                     ++bin.tangential_pos_num())
                  {
                    int ra = 0, a = 0;
                    int rb = 0, b = 0;

                    proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);

                    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()] = gap_value;


                    proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);
                    int a_in_block = a % num_transaxial_crystals_per_block;

                    if (a_in_block >= num_physical_transaxial_crystals_per_block)
                        continue;
                    int new_a = a - (a / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int ra_in_block = ra % num_axial_crystals_per_block;
                    if (ra_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_ra = ra - (ra / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;

                    int b_in_block = b % num_transaxial_crystals_per_block;
                    if (b_in_block >= num_physical_transaxial_crystals_per_block)
                        continue;
                    int new_b = b - (b / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int rb_in_block = rb % num_axial_crystals_per_block;
                    if (rb_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_rb = rb - (rb / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;

                    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()] =
                    fan_data(new_ra, new_a, new_rb, new_b);
                  }
            proj_data.set_segment(*segment_ptr);
          }
      }
    else
      {
	shared_ptr<const ProjDataInfoCylindricalNoArcCorr> proj_data_info_ptr =
	  get_fan_info(num_rings, num_detectors_per_ring, max_delta, fan_size,
		       *proj_data.get_proj_data_info_sptr());

	const int half_fan_size = fan_size/2;
	//assert(num_rings == fan_data.get_num_rings());
	//assert(num_detectors_per_ring == fan_data.get_num_detectors_per_ring());


	const int num_virtual_axial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_virtual_axial_crystals_per_block();
	const int num_virtual_transaxial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_virtual_transaxial_crystals_per_block();
	const int num_transaxial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_transaxial_crystals_per_block();
	const int num_axial_crystals_per_block =
	  proj_data_info_ptr->get_scanner_sptr()->
	  get_num_axial_crystals_per_block();

	const int num_physical_transaxial_crystals_per_block = num_transaxial_crystals_per_block - num_virtual_transaxial_crystals_per_block;

	const int num_physical_axial_crystals_per_block = num_axial_crystals_per_block - num_virtual_axial_crystals_per_block;

	Bin bin;
	shared_ptr<SegmentBySinogram<float> > segment_ptr;

	for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
	  {
	    segment_ptr.reset(new SegmentBySinogram<float>(proj_data.get_empty_segment_by_sinogram(bin.segment_num())));

	    for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
		 bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
		 ++bin.axial_pos_num())
	      for (bin.view_num() = 0; bin.view_num() < num_detectors_per_ring/2; bin.view_num()++)
                for (bin.tangential_pos_num() = -half_fan_size;
                     bin.tangential_pos_num() <= half_fan_size;
                     ++bin.tangential_pos_num())
		  {
                    int ra = 0, a = 0;
                    int rb = 0, b = 0;

                    proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);

                    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()] = gap_value;


                    proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);
                    int a_in_block = a % num_transaxial_crystals_per_block;

                    if (a_in_block >= num_physical_transaxial_crystals_per_block)
                        continue;
                    int new_a = a - (a / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int ra_in_block = ra % num_axial_crystals_per_block;
                    if (ra_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_ra = ra - (ra / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;

                    int b_in_block = b % num_transaxial_crystals_per_block;
                    if (b_in_block >= num_physical_transaxial_crystals_per_block)
                        continue;
                    int new_b = b - (b / num_transaxial_crystals_per_block) * num_virtual_transaxial_crystals_per_block;

                    int rb_in_block = rb % num_axial_crystals_per_block;
                    if (rb_in_block >= num_physical_axial_crystals_per_block)
                        continue;
                    int new_rb = rb - (rb / num_axial_crystals_per_block) * num_virtual_axial_crystals_per_block;
                    
                    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()] =
                    fan_data(new_ra, new_a, new_rb, new_b);
		  }
	    proj_data.set_segment(*segment_ptr);
	  }
      }
}


void apply_block_norm(FanProjData& fan_data, const BlockData3D& block_data, const bool apply)
{
  const int num_axial_detectors = fan_data.get_num_rings();
  const int num_tangential_detectors = fan_data.get_num_detectors_per_ring();
  const int num_axial_blocks = block_data.get_num_rings();
  const int num_tangential_blocks = block_data.get_num_detectors_per_ring();
  const int num_axial_crystals_per_block = num_axial_detectors/num_axial_blocks;
  assert(num_axial_blocks * num_axial_crystals_per_block == num_axial_detectors);
  const int num_tangential_crystals_per_block = num_tangential_detectors/num_tangential_blocks;
  assert(num_tangential_blocks * num_tangential_crystals_per_block == num_tangential_detectors);
  
  for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      // loop rb from ra to avoid double counting
      for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
        for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)      
      {
	// note: add 2*num_detectors_per_ring to newb to avoid using mod with negative numbers
	if (fan_data(ra,a,rb,b) == 0)
	  continue;
        if (apply)
          fan_data(ra,a,rb,b) *=
	    block_data(ra/num_axial_crystals_per_block,a/num_tangential_crystals_per_block,
                       rb/num_axial_crystals_per_block,b/num_tangential_crystals_per_block);
        else
          fan_data(ra,a,rb,b) /=
	    block_data(ra/num_axial_crystals_per_block,a/num_tangential_crystals_per_block,
                       rb/num_axial_crystals_per_block,b/num_tangential_crystals_per_block);
      }
}

#if 0

void apply_geo_norm(FanProjData& fan_data, const GeoData& geo_data, const bool apply)
{
  const int num_detectors_per_ring = fan_data.get_num_detectors_per_ring();
  const int num_crystals_per_block = geo_data.get_length()*2;

  for (int a = fan_data.get_min_ra(); a <= fan_data.get_max_ra(); ++a)
    for (int b = fan_data.get_min_ra(a); b <= fan_data.get_max_ra(a); ++b)      
      {
	if (fan_data(ra,a,rb,b) == 0)
	  continue;
        int newa = a % num_crystals_per_block;
	int newb = b - (a - newa); 
	if (newa > num_crystals_per_block - 1 - newa)
	  { 
	    newa = num_crystals_per_block - 1 - newa; 
	    newb = - newb + num_crystals_per_block - 1;
	  }
	// note: add 2*num_detectors_per_ring to newb to avoid using mod with negative numbers
        if (apply)
          fan_data(ra,a,rb,b) *=
	    geo_data[newa][(2*num_detectors_per_ring + newb)%num_detectors_per_ring];
        else
          fan_data(ra,a,rb,b) /=
	    geo_data[newa][(2*num_detectors_per_ring + newb)%num_detectors_per_ring];
      }
}
#endif

void apply_geo_norm(FanProjData& fan_data, const GeoData3D& geo_data, const bool apply)
{
    
    const int num_axial_detectors = fan_data.get_num_rings();
    const int num_transaxial_detectors = fan_data.get_num_detectors_per_ring();
    const int num_axial_crystals_per_block = geo_data.get_num_axial_crystals_per_block();
    const int num_transaxial_crystals_per_block = geo_data.get_half_num_transaxial_crystals_per_block()*2;
    
    const int num_transaxial_blocks = num_transaxial_detectors/num_transaxial_crystals_per_block;
    const int num_axial_blocks = num_axial_detectors/num_axial_crystals_per_block;
    
    FanProjData work = fan_data;
    work.fill(0);
    
    
    for (int ra = 0; ra < num_axial_crystals_per_block; ++ra)
        for (int a = 0; a < num_transaxial_crystals_per_block /2; ++a)
            // loop rb from ra to avoid double counting
            for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
                for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)
                {
                    
                    
                    // rotation
                    
                    for (int axial_block_num = 0; axial_block_num<num_axial_blocks; ++axial_block_num)
                    {
                        
                        for (int transaxial_block_num = 0; transaxial_block_num<num_transaxial_blocks; ++transaxial_block_num)
                        {
                            
                            
                            const int transaxial_det_inc = transaxial_block_num * num_transaxial_crystals_per_block;
                            const int new_det_num_a = (a+transaxial_det_inc)%num_transaxial_detectors;
                            const int new_det_num_b = (b+transaxial_det_inc)%num_transaxial_detectors;
                            const int axial_det_inc = axial_block_num * num_axial_crystals_per_block;
                            const int new_ring_num_a = ra+axial_det_inc;
                            const int new_ring_num_b = rb+axial_det_inc;
                            
                            const int ma = num_transaxial_detectors-1-new_det_num_a;
                            const int mb = (2*num_transaxial_detectors-1-new_det_num_b)%num_transaxial_detectors;
                            const int mra = num_axial_detectors-1-new_ring_num_a;
                            const int mrb = num_axial_detectors-1-new_ring_num_b;
                            
                            if (work.is_in_data(new_ring_num_a,new_det_num_a,new_ring_num_b,new_det_num_b))
                                work(new_ring_num_a, new_det_num_a,new_ring_num_b, new_det_num_b) =
                                geo_data(ra,a,rb,b%num_transaxial_detectors);
                            
                            if (work.is_in_data(new_ring_num_a,ma,new_ring_num_b,mb))
                                work(new_ring_num_a,ma,new_ring_num_b,mb) = geo_data(ra,a,rb,b%num_transaxial_detectors);
                            
                            if (work.is_in_data(mra,new_det_num_a,mrb,new_det_num_b))
                                
                                work(mra,new_det_num_a,mrb,new_det_num_b) = geo_data(ra,a,rb,b%num_transaxial_detectors);
                            
                            if (work.is_in_data(mra,ma,mrb,mb))
                                
                                work(mra,ma,mrb,mb) = geo_data(ra,a,rb,b%num_transaxial_detectors);
                            
                            
                            
                        }
                        
                    }
                }
    
    for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
        for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
        //    for (int rb = fan_data.get_min_ra(); rb <= fan_data.get_max_ra(); ++rb)
	     for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
                for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)
                {
                    
                    if (fan_data(ra,a,rb,b) == 0)
                        continue;
                    if (apply)
                        fan_data(ra,a,rb,b) *= work(ra,a,rb,b%num_transaxial_detectors);
                    else
                        fan_data(ra,a,rb,b) /= work(ra,a,rb,b%num_transaxial_detectors);
                }
    
}


void apply_efficiencies(FanProjData& fan_data, const DetectorEfficiencies& efficiencies, const bool apply)
{
  const int num_detectors_per_ring = fan_data.get_num_detectors_per_ring();
  for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      // loop rb from ra to avoid double counting
      for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
        for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)      
        {
	  if (fan_data(ra,a,rb,b) == 0)
	    continue;
          if (apply)
	    fan_data(ra,a,rb,b) *=
	      efficiencies[ra][a]*efficiencies[rb][b%num_detectors_per_ring];
          else
            fan_data(ra,a,rb,b) /=
	      efficiencies[ra][a]*efficiencies[rb][b%num_detectors_per_ring];
      }
}

void make_fan_sum_data(Array<2,float>& data_fan_sums, const FanProjData& fan_data)
{
  for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      data_fan_sums[ra][a] = fan_data.sum(ra,a);
}

void make_fan_sum_data(Array<2,float>& data_fan_sums,
		       const ProjData& proj_data)
{
  int num_rings;
  int num_detectors_per_ring;
  int fan_size;
  int max_delta;
  if(proj_data.get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry()== "BlocksOnCylindrical")
    {
      shared_ptr<const ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_ptr =
        get_fan_info_block(num_rings, num_detectors_per_ring, max_delta, fan_size,
                     *proj_data.get_proj_data_info_sptr());
      const int half_fan_size = fan_size/2;
      data_fan_sums.fill(0);

      shared_ptr<SegmentBySinogram<float> > segment_ptr;
      Bin bin;

      for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
        {
          segment_ptr.reset(new SegmentBySinogram<float>(proj_data.get_segment_by_sinogram(bin.segment_num())));

          for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
               bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
               ++bin.axial_pos_num())
            for (bin.view_num() = 0; bin.view_num() < num_detectors_per_ring/2; bin.view_num()++)
              for (bin.tangential_pos_num() = -half_fan_size;
                   bin.tangential_pos_num() <= half_fan_size;
                   ++bin.tangential_pos_num())
                {
                  int ra = 0, a = 0;
                  int rb = 0, b = 0;

	          proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);

                  const float value =
                    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()];
                  data_fan_sums[ra][a] += value;
                  data_fan_sums[rb][b] += value;
                }
        }      
    }
  else
    {
      shared_ptr<const ProjDataInfoCylindricalNoArcCorr> proj_data_info_ptr =
	get_fan_info(num_rings, num_detectors_per_ring, max_delta, fan_size, 
		     *proj_data.get_proj_data_info_sptr());
      const int half_fan_size = fan_size/2;
      data_fan_sums.fill(0);

      shared_ptr<SegmentBySinogram<float> > segment_ptr;      
      Bin bin;

      for (bin.segment_num() = proj_data.get_min_segment_num(); bin.segment_num() <= proj_data.get_max_segment_num();  ++ bin.segment_num())
	{
	  segment_ptr.reset(new SegmentBySinogram<float>(proj_data.get_segment_by_sinogram(bin.segment_num())));
	  
	  for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
	       bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
	       ++bin.axial_pos_num())
	    for (bin.view_num() = 0; bin.view_num() < num_detectors_per_ring/2; bin.view_num()++)
	      for (bin.tangential_pos_num() = -half_fan_size;
		   bin.tangential_pos_num() <= half_fan_size;
		   ++bin.tangential_pos_num())
		{
		  int ra = 0, a = 0;
		  int rb = 0, b = 0;
            
		  proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, bin);

		  const float value =            
		    (*segment_ptr)[bin.axial_pos_num()][bin.view_num()][bin.tangential_pos_num()];
		  data_fan_sums[ra][a] += value;
		  data_fan_sums[rb][b] += value;
		}
	}
    }
}

void make_fan_sum_data(Array<2,float>& data_fan_sums,
		       const DetectorEfficiencies& efficiencies,
		       const int max_ring_diff, const int half_fan_size)
{
  const int num_rings = data_fan_sums.get_length();
  assert(data_fan_sums.get_min_index()==0);
  const int num_detectors_per_ring = 
    data_fan_sums[0].get_length();

  for (int ra = data_fan_sums.get_min_index(); ra <= data_fan_sums.get_max_index(); ++ra)
    for (int a = data_fan_sums[ra].get_min_index(); a <= data_fan_sums[ra].get_max_index(); ++a)
      {
	float fan_sum = 0;
	for (int rb = max(ra-max_ring_diff, 0); rb <= min(ra+max_ring_diff, num_rings-1); ++rb)
	  for (int b = a+num_detectors_per_ring/2-half_fan_size; b <= a+num_detectors_per_ring/2+half_fan_size; ++b)
	    fan_sum += efficiencies[rb][b%num_detectors_per_ring];
	data_fan_sums[ra][a] = efficiencies[ra][a]*fan_sum;
      }
}


void make_geo_data(GeoData3D& geo_data, const FanProjData& fan_data)
{
    
    const int num_axial_detectors = fan_data.get_num_rings();
    const int num_transaxial_detectors = fan_data.get_num_detectors_per_ring();
    const int num_axial_crystals_per_block = geo_data.get_num_axial_crystals_per_block();
    const int num_transaxial_crystals_per_block = geo_data.get_half_num_transaxial_crystals_per_block()*2;
    const int num_transaxial_blocks = num_transaxial_detectors/num_transaxial_crystals_per_block;
    const int num_axial_blocks = num_axial_detectors/num_axial_crystals_per_block;
    
    
    // transaxial and axial mirroring
    
    FanProjData work = fan_data;
    work.fill(0);
    
    for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
        for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
           //1// for (int rb = fan_data.get_min_ra(); rb <= fan_data.get_max_ra(); ++rb)
            for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
                for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)
                {
                    
                    const int ma = num_transaxial_detectors-1-a;
                    const int mb = (2*num_transaxial_detectors-1-b)%num_transaxial_detectors;
                    const int mra = num_axial_detectors-1-ra;
                    const int mrb = (num_axial_detectors-1-rb);
                    
                    
                    if (ra!=mra && rb !=mrb)
                        work(ra,a,rb,b)= fan_data(ra,a,rb,b) + fan_data(ra,ma,rb,mb) + fan_data(mra,a,mrb,b) + fan_data(mra,ma,mrb,mb);
                    else
                        work(ra,a,rb,b)= fan_data(ra,a,rb,b) + fan_data(ra,ma,rb,mb);
                    
                }
    
    
    geo_data.fill(0);
    
    
    for (int ra = 0; ra < num_axial_crystals_per_block; ++ra)
      //  for (int a = 0; a <= num_transaxial_detectors/2; ++a)
        for (int a = 0; a <num_transaxial_crystals_per_block/2; ++a)
            // loop rb from ra to avoid double counting
           // for (int rb = fan_data.get_min_ra(); rb <= fan_data.get_max_ra(); ++rb)
	     for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
                for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)
                {
                    
                    
                    // rotation
                    
                    for (int axial_block_num = 0; axial_block_num<num_axial_blocks; ++axial_block_num)
                    {
                        
                        for (int transaxial_block_num = 0; transaxial_block_num<num_transaxial_blocks; ++transaxial_block_num)
                        {
                            
                            const int transaxial_det_inc = transaxial_block_num * num_transaxial_crystals_per_block;
                            const int new_det_num_a = (a+transaxial_det_inc)%num_transaxial_detectors;
                            const int new_det_num_b = (b+transaxial_det_inc)%num_transaxial_detectors;
                            const int axial_det_inc = axial_block_num * num_axial_crystals_per_block;
                            const int new_ring_num_a = ra+axial_det_inc;
                            const int new_ring_num_b = rb+axial_det_inc;
                            if (fan_data.is_in_data(new_ring_num_a,new_det_num_a,new_ring_num_b,new_det_num_b))
                                geo_data(ra,a,rb,b%num_transaxial_detectors) +=
                                work(new_ring_num_a, new_det_num_a,new_ring_num_b, new_det_num_b);
                                
                                }
                        
                    }
                }
}


void make_block_data(BlockData3D& block_data, const FanProjData& fan_data)
{
  const int num_axial_detectors = fan_data.get_num_rings();
  const int num_transaxial_detectors = fan_data.get_num_detectors_per_ring();
  const int num_axial_blocks = block_data.get_num_rings();
  const int num_transaxial_blocks = block_data.get_num_detectors_per_ring();
  const int num_axial_crystals_per_block = num_axial_detectors/num_axial_blocks;
  assert(num_axial_blocks * num_axial_crystals_per_block == num_axial_detectors);
  const int num_transaxial_crystals_per_block = num_transaxial_detectors/num_transaxial_blocks;
  assert(num_transaxial_blocks * num_transaxial_crystals_per_block == num_transaxial_detectors);
  
  block_data.fill(0);
  for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      // loop rb from ra to avoid double counting
      for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
        for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)      
        {
          block_data(ra/num_axial_crystals_per_block,a/num_transaxial_crystals_per_block,
                     rb/num_axial_crystals_per_block,b/num_transaxial_crystals_per_block) +=
	  fan_data(ra,a,rb,b);
        }  
}

void iterate_efficiencies(DetectorEfficiencies& efficiencies,
			  const Array<2,float>& data_fan_sums,
			  const FanProjData& model)
{
  const int num_detectors_per_ring = model.get_num_detectors_per_ring();
  assert(model.get_min_ra() == data_fan_sums.get_min_index());
  assert(model.get_max_ra() == data_fan_sums.get_max_index());
  assert(model.get_min_a() == data_fan_sums[data_fan_sums.get_min_index()].get_min_index());
  assert(model.get_max_a() == data_fan_sums[data_fan_sums.get_min_index()].get_max_index());
  for (int ra = model.get_min_ra(); ra <= model.get_max_ra(); ++ra)
    for (int a = model.get_min_a(); a <= model.get_max_a(); ++a)
    {
      if (data_fan_sums[ra][a] == 0)
	efficiencies[ra][a] = 0;
      else
	{
     	  float denominator = 0;
           for (int rb = model.get_min_rb(ra); rb <= model.get_max_rb(ra); ++rb)
             for (int b = model.get_min_b(a); b <= model.get_max_b(a); ++b)
  	       denominator += efficiencies[rb][b%num_detectors_per_ring]*model(ra,a,rb,b);
	  efficiencies[ra][a] = data_fan_sums[ra][a] / denominator;
	}
    }
}

// version without model
void iterate_efficiencies(DetectorEfficiencies& efficiencies,
			  const Array<2,float>& data_fan_sums,
			  const int max_ring_diff, const int half_fan_size)
{
  const int num_rings = data_fan_sums.get_length();
  const int num_detectors_per_ring = data_fan_sums[data_fan_sums.get_min_index()].get_length();
#ifdef WRITE_ALL
  static int sub_iter_num = 0;
#endif
  for (int ra = data_fan_sums.get_min_index(); ra <= data_fan_sums.get_max_index(); ++ra)
    for (int a = data_fan_sums[ra].get_min_index(); a <= data_fan_sums[ra].get_max_index(); ++a)
    {
      if (data_fan_sums[ra][a] == 0)
	efficiencies[ra][a] = 0;
      else
	{
     	  float denominator = 0;
	  for (int rb = max(ra-max_ring_diff, 0); rb <= min(ra+max_ring_diff, num_rings-1); ++rb)
             for (int b = a+num_detectors_per_ring/2-half_fan_size; b <= a+num_detectors_per_ring/2+half_fan_size; ++b)
  	       denominator += efficiencies[rb][b%num_detectors_per_ring];
	  efficiencies[ra][a] = data_fan_sums[ra][a] / denominator;
	}
#ifdef WRITE_ALL
          {
            char out_filename[100];
            sprintf(out_filename, "MLresult_subiter_eff_1_%d.out", 
		    sub_iter_num++);
            ofstream out(out_filename);
	    if (!out)
	      {
		warning("Error opening output file %s\n", out_filename);
		exit(EXIT_FAILURE);
	      }
            out << efficiencies;
	    if (!out)
	      {
		warning("Error writing data to output file %s\n", out_filename);
		exit(EXIT_FAILURE);
	      }
          }
#endif
    }
}

void iterate_geo_norm(GeoData3D& norm_geo_data,
                      const GeoData3D& measured_geo_data,
                      const FanProjData& model)
{
    make_geo_data(norm_geo_data, model);
    //norm_geo_data = measured_geo_data / norm_geo_data;
    
    const int num_axial_crystals_per_block = measured_geo_data.get_num_axial_crystals_per_block();
    const int num_transaxial_crystals_per_block = measured_geo_data.get_half_num_transaxial_crystals_per_block()*2;
    
    const float threshold = measured_geo_data.find_max()/10000.F;
    
    for (int ra = 0; ra < num_axial_crystals_per_block; ++ra)
        for (int a = 0; a <num_transaxial_crystals_per_block/2; ++a)
            // loop rb from ra to avoid double counting
         //   for (int rb = model.get_min_ra(); rb <= model.get_max_ra(); ++rb)
	     for (int rb = max(ra,model.get_min_rb(ra)); rb <= model.get_max_rb(ra); ++rb)
                for (int b =model.get_min_b(a); b <= model.get_max_b(a); ++b)
                {
                    
                    norm_geo_data(ra,a,rb,b) = (measured_geo_data(ra,a,rb,b)>=threshold || measured_geo_data(ra,a,rb,b) < 10000*norm_geo_data(ra,a,rb,b))? measured_geo_data(ra,a,rb,b) / norm_geo_data(ra,a,rb,b)
                    : 0;
                }
}

void iterate_block_norm(BlockData3D& norm_block_data,
		      const BlockData3D& measured_block_data,
		      const FanProjData& model)
{
  make_block_data(norm_block_data, model);
  //norm_block_data = measured_block_data / norm_block_data;
  const float threshold = measured_block_data.find_max()/10000.F;
  for (int ra = norm_block_data.get_min_ra(); ra <= norm_block_data.get_max_ra(); ++ra)
    for (int a = norm_block_data.get_min_a(); a <= norm_block_data.get_max_a(); ++a)
      // loop rb from ra to avoid double counting
      for (int rb = max(ra,norm_block_data.get_min_rb(ra)); rb <= norm_block_data.get_max_rb(ra); ++rb)
        for (int b = norm_block_data.get_min_b(a); b <= norm_block_data.get_max_b(a); ++b)      
      {
	norm_block_data(ra,a,rb,b) =
	  (measured_block_data(ra,a,rb,b)>=threshold ||
	   measured_block_data(ra,a,rb,b) < 10000*norm_block_data(ra,a,rb,b))
	  ? measured_block_data(ra,a,rb,b) / norm_block_data(ra,a,rb,b)
	  : 0;
      }
}


double KL(const FanProjData& d1, const FanProjData& d2, const double threshold)
{
  double sum=0;
  for (int ra = d1.get_min_ra(); ra <= d1.get_max_ra(); ++ra)
    {
      double asum=0;
      for (int a = d1.get_min_a(); a <= d1.get_max_a(); ++a)
        {
          double rbsum=0;
          for (int rb = max(ra,d1.get_min_rb(ra)); rb <= d1.get_max_rb(ra); ++rb)
            {
              double bsum=0;
              for (int b = d1.get_min_b(a); b <= d1.get_max_b(a); ++b)      
                bsum += static_cast<double>(KL(d1(ra,a,rb,b), d2(ra,a,rb,b), threshold));
              rbsum += bsum;
            }
          asum += rbsum;
        }
      sum += asum;
    }
  return static_cast<double>(sum);
}

/*double KL(const GeoData3D& d1, const GeoData3D& d2, const double threshold)
{
    const int num_axial_detectors = d1.get_num_rings();
    const int num_transaxial_detectors = d1.get_num_detectors_per_ring();
    const int num_axial_crystals_per_block = d1.get_num_axial_crystals_per_block();
    const int num_transaxial_crystals_per_block = d1.get_half_num_transaxial_crystals_per_block()*2;
     double sum=0;
    
    for (int ra = 0; ra < num_axial_crystals_per_block; ++ra)
	{
       		double asum=0;
      		 for (int a = 0; a <num_transaxial_crystals_per_block/2; ++a)
		{
                 double rbsum=0;
	         for (int rb = max(ra,d1.get_min_rb(ra)); rb <= d1.get_max_rb(ra); ++rb)
		{
	              double bsum=0;
              
		  for (int b =d1.get_min_b(a); b <= d1.get_max_b(a); ++b)
                
                bsum += static_cast<double>(KL(d1(ra,a,rb,b), d2(ra,a,rb,b), threshold));
              rbsum += bsum;
            }
          asum += rbsum;
        }
      sum += asum;
    }
  return static_cast<double>(sum);
}
*/
                   

END_NAMESPACE_STIR