//
//
/*!
  \file
  \ingroup symmetries

  \brief non-inline implementations for all symmetry operations for PET scanners
  and Cartesian grid images

  This file contains just tedious repetitions of the default
  implementation of SymmetryOperation::transform_proj_matrix_elems_for_one_bin,
  with 1 tiny (but important) change: the call to transform_image_coordinates
  is explicitly forced to the member of the class (instead of just calling
  the virtual function). The result of this is that the compiler can
  inline the function, and hence dramatic speed-up.

  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
#include "stir/recon_buildblock/SymmetryOperations_PET_CartesianGrid.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/Coordinate3D.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.h"


START_NAMESPACE_STIR

void
SymmetryOperation_PET_CartesianGrid_z_shift::
transform_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  Bin bin = lor.get_bin();
  transform_bin_coordinates(bin);
  lor.set_bin(bin);

  ProjMatrixElemsForOneBin::iterator element_ptr = lor.begin();
  while (element_ptr != lor.end()) 
  {
    // TODO possibly an explicit z_shift here would be quicker, although a smart compiler should see it
    Coordinate3D<int> c(element_ptr->get_coords());
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_z_shift::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmx_zq::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xy_yx::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmx::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xmx::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_ymy::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_ymy::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_zq::
transform_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  ProjMatrixElemsForOneBin::iterator element_ptr = lor.begin();
  while (element_ptr != lor.end()) 
  {
    Coordinate3D<int> c(element_ptr->get_coords());
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_ymy_zq::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}


void 
SymmetryOperation_PET_CartesianGrid_swap_ymy_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}

void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


void
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq::
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
    self::transform_image_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value());
    ++element_ptr;
  }
}



void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq::
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
    self::transform_bin_coordinates(c);
    *element_ptr = ProjMatrixElemsForOneDensel::value_type(c);
    ++element_ptr;
  }
} 


END_NAMESPACE_STIR
