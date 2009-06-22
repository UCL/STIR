//
// $Id$
//
/*!

  \file
  \ingroup projdata

  \brief Implementation of inline functions of class stir::ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

#include "stir/Bin.h"

START_NAMESPACE_STIR

float
ProjDataInfoCylindricalArcCorr::get_s(const Bin& bin) const
{return bin.tangential_pos_num()*bin_size;}


float
ProjDataInfoCylindricalArcCorr::get_tangential_sampling() const
{return bin_size;}

END_NAMESPACE_STIR

