//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    Some parts of this file originate in CTI code, distributed as
    part of the matrix library from Louvain-la-Neuve, and hence carries
    its restrictive license. Affected parts are the dead-time correction
    in get_deadtime_efficiency and geo_Z_corr related code.

    Most of this file is free software; you can redistribute that part and/or modify
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
  \ingroup modelling
  \brief File that registers all RegisterObject children in modelling

  \author Charalampos Tsoumpas
  
  $Date$
  $Revision$
*/
#include "local/stir/modelling/PatlakPlot.h"
//#include "local/stir/modelling/KineticModel.h"

START_NAMESPACE_STIR

static PatlakPlot::RegisterIt dummy113;
//static KineticModel::RegisterIt dummy213; // ChT::It seems to work fine without it. I have not understand yet why.

END_NAMESPACE_STIR

