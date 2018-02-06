/*
 Copyright (C) 2009- 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */  
/*!
 \file
 \ingroup GeneralisedObjectiveFunction
 \brief Instantiations for class stir::PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion
 \author Charalampos Tsoumpas

*/

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.txx"

START_NAMESPACE_STIR

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif // _MSC_VER
template class 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<DiscretisedDensity<3,float> >;

END_NAMESPACE_STIR
