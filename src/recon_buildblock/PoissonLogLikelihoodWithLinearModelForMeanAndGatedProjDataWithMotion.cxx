/*
 Copyright (C) 2009- 2013, King's College London
 This file is part of STIR.
 
 SPDX-License-Identifier: Apache-2.0
 
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
