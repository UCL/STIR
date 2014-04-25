//
// $Id: PoissonNestedLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.cxx,v 1.0 2013-07-12 10:34:00 kris Exp $
//
/*
  Copyright (C) 2006- $Date: 2013-07-12 10:34:00 $, Hammersmith Imanet Ltd
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
  \ingroup GeneralisedObjectiveFunction
  \brief Instantiations for class stir::PoissonNestedLogLikelihoodWithLinearKineticModelAndDynamicProjectionData

  \author Nicolas A Karakatsanis

  $Date: 2013-12-07 10:34:00 $
  $Revision: 1.0 $
*/

#include "stir/recon_buildblock/PoissonNestedLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.txx"

START_NAMESPACE_STIR

#ifdef _MSC_VER
// prevent warning message on instantiation of abstract class
#  pragma warning(disable : 4661)
#endif // _MSC_VER

template class PoissonNestedLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<ParametricVoxelsOnCartesianGrid>;

END_NAMESPACE_STIR
