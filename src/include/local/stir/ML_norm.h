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

#define ARRAY_CONST_IT  

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

END_NAMESPACE_STIR
#endif
