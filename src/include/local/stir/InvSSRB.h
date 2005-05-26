//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \ingroup projdata
  \brief Declaration of InvSSRB functions

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SSRB_H__
#define __stir_SSRB_H__

#include "stir/common.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

class ProjData;
class ProjDataInfo;

//! Perform Inverse Single Slice Rebinning and write output to ProjData3D format
/*! 
  \ingroup projdata
  \param proj_data_3D output projection data. Its projection_data_info is used to 
  determine output characteristics. Data will be 'put' in here using 
  ProjData::set_sinogram().
  \param proj_data_2D input data

  The STIR implementation of Inverse SSRB is a generalisation that applies the 
  inverse idea of the SSRB. For instance, for a 2D dataset, Inverse SSRB can 
  produce a new expanded 3D dataset based on the given information proj_data_3D.
  This implementation preserves the equation: 
					proj_data_2D=SSRB(InvSSRB(proj_data_2D)
  This mostly useful in the scatter sinogram expansion, see the STIR Glossary on 
  scatter correction. 
  
  \warning in_proj_data_info has to be (at least) of type ProjDataInfoCylindrical
*/  
void 
InvSSRB(ProjData& proj_data_3D,
		const ProjData& proj_data_2D);

END_NAMESPACE_STIR

#endif

