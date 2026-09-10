//
/*
  Copyright (C) 2006- $Date: 2013-07-12 10:34:00 $, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Instantiations for class stir::PoissonNestedLogLikelihoodWithLinearKineticModelAndDynamicProjectionData

  \author Nicolas A Karakatsanis

*/

#include "stir/recon_buildblock/PoissonNestedLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.txx"

START_NAMESPACE_STIR

#ifdef _MSC_VER
// prevent warning message on instantiation of abstract class
#  pragma warning(disable : 4661)
#endif // _MSC_VER

template class PoissonNestedLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<ParametricVoxelsOnCartesianGrid>;

END_NAMESPACE_STIR
