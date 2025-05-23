//
//
/*
  Copyright (C) 2006- 2009, Hammersmith Imanet Ltd
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Instantiations for class stir::PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.txx"

START_NAMESPACE_STIR


#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif // _MSC_VER

template class PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<ParametricVoxelsOnCartesianGrid >;

END_NAMESPACE_STIR
