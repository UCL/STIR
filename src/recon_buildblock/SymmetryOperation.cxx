//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementations of non-inline functions for class SymmetryOperation

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "recon_buildblock/SymmetryOperation.h"
#include "recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "Coordinate3D.h"

START_NAMESPACE_TOMO

void 
SymmetryOperation::transform_proj_matrix_elems_for_one_bin(
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
};

END_NAMESPACE_TOMO
