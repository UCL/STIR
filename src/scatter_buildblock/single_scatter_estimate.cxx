//
//
/*Copyright (C) 2004- 2009, Hammersmith Imanet
  Copyright (C) 2016, UCL
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
  \ingroup scatter
  \brief Implementations of stir::ScatterEstimationByBin::scatter_estimate and stir::ScatterEstimationByBin::single_scatter_estimate

  \author Nikos Efthimiou
  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans

*/
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/scatter/ScatterEstimation.h"
using namespace std;
START_NAMESPACE_STIR



double
ScatterSimulation::
scatter_estimate(const unsigned det_num_A, 
		 const unsigned det_num_B)	
{
  double scatter_ratio_singles = 0;

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
}

END_NAMESPACE_STIR
