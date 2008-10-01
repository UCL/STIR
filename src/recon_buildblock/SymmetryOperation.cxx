//
// $Id$
//
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
*//*!
  \file
  \ingroup symmetries

  \brief Implementations of non-inline functions for class SymmetryOperation

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/

#include "stir/recon_buildblock/SymmetryOperation.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/Coordinate3D.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.h"

START_NAMESPACE_STIR

void 
SymmetryOperation::
transform_proj_matrix_elems_for_one_bin(
					ProjMatrixElemsForOneBin& lor) const 
{
  Bin bin = lor.get_bin();
  transform_bin_coordinates(bin);
  lor.set_bin(bin);
  
  ProjMatrixElemsForOneBin::iterator element_ptr = lor.begin();
  while (element_ptr != lor.end()) 
  {
    Coordinate3D<int> c(element_ptr->get_coords());
    transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation::
transform_proj_matrix_elems_for_one_densel(
				       ProjMatrixElemsForOneDensel& probs) const
{
  Densel densel = probs.get_densel();
  transform_image_coordinates(densel);
  probs.set_densel(densel);
  
  ProjMatrixElemsForOneDensel::iterator element_ptr = probs.begin();
  while (element_ptr != probs.end()) 
  {
    Bin c(*element_ptr);
    transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


END_NAMESPACE_STIR
