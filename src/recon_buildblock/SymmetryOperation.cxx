//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Implementations of non-inline functions for class SymmetryOperation

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
#include "recon_buildblock/SymmetryOperation.h"
#include "recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "Coordinate3D.h"
#ifdef ENABLE_DENSEL 
#include "local/tomo/recon_buildblock/ProjMatrixElemsForOneDensel.h"
#endif

START_NAMESPACE_TOMO

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


#ifdef ENABLE_DENSEL 
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
#endif

END_NAMESPACE_TOMO
