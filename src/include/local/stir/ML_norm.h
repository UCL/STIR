//
//
/*
    Copyright (C) 2001- 2011, Hammersmith Imanet Ltd
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
  \ingroup buildblock

  \brief Preliminary things for ML normalisation factor estimation

  \author Kris Thielemans

*/
#ifndef __stir_ML_norm_H__
#define __stir_ML_norm_H__


#include "stir/ProjData.h"
#include "stir/Array.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/IndexRange2D.h"
#include "stir/Sinogram.h"
#include <string>
#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::string;
using std::ostream;
using std::istream;
#endif

START_NAMESPACE_STIR

typedef Array<2,float> GeoData;
typedef Array<2,float> BlockData;

class DetPairData : private Array<2,float>
{
public:

  DetPairData();
  DetPairData(const IndexRange<2>& range);
  DetPairData& operator=(const DetPairData&);

  float& operator()(const int a, const int b);
  
  float operator()(const int a, const int b) const;

  void fill(const float d);

  int get_min_index() const;
  int get_max_index() const;
  int get_min_index(const int a) const;
  int get_max_index(const int a) const;
  bool is_in_data(const int a, const int b) const;
  float sum() const;
  float sum(const int a) const;
  float find_max() const;
  float find_min() const;
  int get_num_detectors() const;
  void grow(const IndexRange<2>&);

private:
  typedef Array<2,float> base_type;
  int num_detectors;
};

void display(const DetPairData&,const char * const);

//! Makes a DetPairData of appropriate dimensions and fills it with 0
void make_det_pair_data(DetPairData& det_pair_data,
			const ProjDataInfo& proj_data_info,
			const int segment_num,
			const int ax_pos_num);
void make_det_pair_data(DetPairData& det_pair_data,
			const ProjData& proj_data,
			const int segment_num,
			const int ax_pos_num);
void set_det_pair_data(ProjData& proj_data,
                       const DetPairData& det_pair_data,
			const int segment_num,
			const int ax_pos_num);
void apply_block_norm(DetPairData& det_pair_data, 
                    const BlockData& geo_data, 
                    const bool apply= true);
void apply_geo_norm(DetPairData& det_pair_data, 
                    const GeoData& geo_data, 
                    const bool apply= true);
void apply_efficiencies(DetPairData& det_pair_data, 
                        const Array<1,float>& efficiencies, 
                        const bool apply=true);


void make_fan_sum_data(Array<1,float>& data_fan_sums, const DetPairData& det_pair_data);

void make_geo_data(GeoData& geo_data, const DetPairData& det_pair_data);
 
void make_block_data(BlockData& block_data, const DetPairData& det_pair_data);
 
void iterate_efficiencies(Array<1,float>& efficiencies,
			  const Array<1,float>& data_fan_sums,
			  const DetPairData& model);

void iterate_geo_norm(GeoData& norm_geo_data,
		      const GeoData& measured_geo_data,
		      const DetPairData& model);

void iterate_block_norm(BlockData& norm_block_data,
			const BlockData& measured_block_data,
			const DetPairData& model);

//******3 D
typedef Array<2,float> DetectorEfficiencies;

class FanProjData : private Array<4,float>
{
public:

  FanProjData();
  FanProjData(const int num_rings, const int num_detectors_per_ring, const int max_ring_diff, const int fan_size);
  virtual ~FanProjData();
  FanProjData& operator=(const FanProjData&);

  float& operator()(const int ra, const int a, const int rb, const int b);
  
  float operator()(const int ra, const int a, const int rb, const int b) const;

  void fill(const float d);

  int get_min_ra() const;
  int get_max_ra() const;
  int get_min_a() const;
  int get_max_a() const;
  int get_min_b(const int a) const;
  int get_max_b(const int a) const;
  int get_min_rb(const int ra) const;
  int get_max_rb(const int ra) const;
  bool is_in_data(const int ra, const int a, const int rb, const int b) const;
  float sum() const;
  float sum(const int ra, const int a) const;
  float find_max() const;
  float find_min() const;
  int get_num_detectors_per_ring() const;
  int get_num_rings() const;


private:
  friend ostream& operator<<(ostream&, const FanProjData&);
  friend istream& operator>>(istream&, FanProjData&);
  typedef Array<4,float> base_type;
  //FanProjData(const IndexRange<4>& range);
  //void grow(const IndexRange<4>&);
  int num_rings;
  int num_detectors_per_ring;
  int max_ring_diff;
  int half_fan_size;
};

typedef FanProjData BlockData3D;

void display(const FanProjData&,const char * const);


shared_ptr<ProjDataInfoCylindricalNoArcCorr>
get_fan_info(int& num_rings, int& num_detectors_per_ring, 
	     int& max_ring_diff, int& fan_size, 
	     const ProjDataInfo& proj_data_info);

void make_fan_data(FanProjData& fan_data,
			const ProjData& proj_data);
void set_fan_data(ProjData& proj_data,
                       const FanProjData& fan_data);

void apply_block_norm(FanProjData& fan_data, 
                    const BlockData3D& block_data, 
                    const bool apply= true);
#if 0
void apply_geo_norm(FanProjData& fan_data, 
                    const GeoData& geo_data, 
                    const bool apply= true);
#endif

void apply_efficiencies(FanProjData& fan_data, 
                        const DetectorEfficiencies& efficiencies, 
                        const bool apply=true);

void make_fan_sum_data(Array<2,float>& data_fan_sums, const FanProjData& fan_data);

void make_fan_sum_data(Array<2,float>& data_fan_sums,
		       const ProjData& proj_data);

void make_fan_sum_data(Array<2,float>& data_fan_sums,
		       const DetectorEfficiencies& efficiencies,
		       const int max_ring_diff, const int half_fan_size);


void make_block_data(BlockData3D& block_data, const FanProjData& fan_data);


void iterate_efficiencies(DetectorEfficiencies& efficiencies,
			  const Array<2,float>& data_fan_sums,
			  const FanProjData& model);

// version without model
void iterate_efficiencies(DetectorEfficiencies& efficiencies,
			  const Array<2,float>& data_fan_sums,
			  const int max_ring_diff, const int half_fan_size);

void iterate_block_norm(BlockData3D& norm_block_data,
			const BlockData3D& measured_block_data,
			const FanProjData& model);

inline float KL(const float a, const float b, const float threshold_a = 0)
{
  assert(a>=0);
  assert(b>=0);
  float res = a<=threshold_a ? b : (a*(log(a)-log(b)) + b - a);
#ifndef NDEBUG
  if (res != res)
    warning("KL nan at a=%g b=%g, threshold %g\n",a,b,threshold_a);
  if (res > 1.E20)
    warning("KL large at a=%g b=%g, threshold %g\n",a,b,threshold_a);
#endif
  assert(res>=-1.e-4);
  return res;
}

template <int num_dimensions, typename elemT>
float KL(const Array<num_dimensions, elemT>& a, const Array<num_dimensions, elemT>& b, const float threshold_a = 0)
{
  double sum = 0;
  typename Array<num_dimensions, elemT>::const_full_iterator iter_a = a.begin_all();
  typename Array<num_dimensions, elemT>::const_full_iterator iter_b = b.begin_all();
  while (iter_a != a.end_all())
    {
      sum += static_cast<double>(KL(*iter_a++, *iter_b++, threshold_a));
    }
  return static_cast<float>(sum);
}

float KL(const DetPairData& d1, const DetPairData& d2, const float threshold);

float KL(const FanProjData& d1, const FanProjData& d2, const float threshold);

END_NAMESPACE_STIR
#endif
