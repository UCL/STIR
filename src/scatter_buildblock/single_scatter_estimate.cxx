//
//
/*Copyright (C) 2004- 2009, Hammersmith Imanet
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0 

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup scatter
  \brief Implementation of stir::SingleScatterSimulation::actual_scatter_estimate

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans

*/
#include "stir/scatter/SingleScatterSimulation.h"
START_NAMESPACE_STIR
static const float total_Compton_cross_section_511keV = 
ScatterSimulation::
  total_Compton_cross_section(511.F); 

double
SingleScatterSimulation::
scatter_estimate(const Bin& bin)
{
  double scatter_ratio_singles = 0;
  unsigned det_num_A = 0; // initialise to avoid compiler warnings
  unsigned det_num_B = 0;

  this->find_detectors(det_num_A, det_num_B, bin);

  this->actual_scatter_estimate(scatter_ratio_singles,
				det_num_A,
				det_num_B);

 return scatter_ratio_singles;
}


void
SingleScatterSimulation::
actual_scatter_estimate(double& scatter_ratio_singles,
			const unsigned det_num_A, 
			const unsigned det_num_B)
{

  scatter_ratio_singles = 0;
		
  for(std::size_t scatter_point_num =0;
      scatter_point_num < this->scatt_points_vector.size();
      ++scatter_point_num)
    {	
	scatter_ratio_singles +=
      simulate_for_one_scatter_point(
							scatter_point_num,
							det_num_A, det_num_B);	

    }	

  // we will divide by the effiency of the detector pair for unscattered photons
  // (computed with the same detection model as used in the scatter code)
  // This way, the scatter estimate will correspond to a 'normalised' scatter estimate.

  // there is a scatter_volume factor for every scatter point, as the sum over scatter points
  // is an approximation for the integral over the scatter point.

  // the factors total_Compton_cross_section_511keV should probably be moved to the scatter_computation code
  const double common_factor =
    1/detection_efficiency_no_scatter(det_num_A, det_num_B) *
    scatter_volume/total_Compton_cross_section_511keV;

  scatter_ratio_singles *= common_factor;
}

END_NAMESPACE_STIR
