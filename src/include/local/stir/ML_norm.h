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

START_NAMESPACE_STIR

typedef Array<2,float> DetPairData;
typedef Array<2,float> GeoData;
typedef Array<2,float> BlockData;

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
