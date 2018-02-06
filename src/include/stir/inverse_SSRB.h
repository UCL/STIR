//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
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
/*
  \ingroup projdata
  \file Declaration of stir::inverse_SSRB
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/


#include "stir/common.h"

START_NAMESPACE_STIR

class ProjData;
class Succeeded;
//! Perform Inverse Single Slice Rebinning and write output to ProjData4D format
/*! 
  \ingroup projdata
  \param[out] proj_data_4D Its projection_data_info is used to 
  determine output characteristics (e.g. number of segments). Data will be 'put' in here using 
  ProjData::set_sinogram().
  \param[in] proj_data_3D input data

  The STIR implementation of Inverse SSRB applies the 
  inverse idea of SSRB. inverse_SSRB will produce oblique 
  sinograms by finding the sinogram that has the same 
  'm'-coordinate (i.e. the intersection with the z-axis
  of the central LOR). In addition, if the output sinogram would lie 'half-way'
  2 input sinograms, it will be set to the average of the 2 input sinograms.

  Note that any oblique segments in \a proj_data_3D are currently ignored.

  Input and output projectino data should have the same number of views and tangential positions.
  
*/  
Succeeded 
inverse_SSRB(ProjData& proj_data_4D,
	     const ProjData& proj_data_3D);

END_NAMESPACE_STIR

