//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Preliminary things for ML normalisation factor estimation

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_ML_norm_H__
#define __stir_ML_norm_H__


#include "stir/ProjData.h"
#include "stir/Array.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/IndexRange2D.h"
#include "stir/Sinogram.h"
#include <string>
#ifndef STIR_NO_NAMESPACES
using std::string;
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


//******3 D
typedef Array<2,float> DetectorEfficiencies;

class FanProjData : private Array<4,float>
{
public:

  FanProjData();
  FanProjData(const int num_rings, const int num_detectors, const int max_ring_diff, const int fan_size);
  FanProjData& operator=(const FanProjData&);

  float& operator()(const int ra, const int a, const int rb, const int b);
  
  float operator()(const int ra, const int a, const int rb, const int b) const;

  void fill(const float d);

  int get_min_index() const;
  int get_max_index() const;
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
  int get_num_detectors() const;
  int get_num_rings() const;


private:
  typedef Array<4,float> base_type;
  FanProjData(const IndexRange<4>& range);
  void grow(const IndexRange<4>&);
  int num_rings;
  int num_detectors;
};

void display(const FanProjData&,const char * const);

void make_fan_data(FanProjData& fan_data,
			const ProjData& proj_data);
void set_fan_data(ProjData& proj_data,
                       const FanProjData& fan_data);
#if 0
void apply_block_norm(FanProjData& fan_data, 
                    const BlockData& geo_data, 
                    const bool apply= true);
void apply_geo_norm(FanProjData& fan_data, 
                    const GeoData& geo_data, 
                    const bool apply= true);
#endif
void apply_efficiencies(FanProjData& fan_data, 
                        const DetectorEfficiencies& efficiencies, 
                        const bool apply=true);



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
  float sum = 0;
  Array<num_dimensions, elemT>::const_full_iterator iter_a = a.begin_all();
  Array<num_dimensions, elemT>::const_full_iterator iter_b = b.begin_all();
  while (iter_a != a.end_all())
    {
      sum += KL(*iter_a++, *iter_b++, threshold_a);
    }
  return sum;
}

END_NAMESPACE_STIR
#endif
