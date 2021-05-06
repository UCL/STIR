//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*//*!
  \file
  \ingroup symmetries

  \brief Implementations of non-inline functions for class SymmetryOperation

  \author Kris Thielemans
  \author PARAPET project

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
