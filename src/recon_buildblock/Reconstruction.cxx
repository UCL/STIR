//
// $Id$
//

/*!

  \file
  \ingroup recon_buildblock
  
  \brief  implementation of the Reconstruction class 
    
  \author Kris Thielemans
      
  $Date$       
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/recon_buildblock/ReconstructionParameters.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/interfile.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

DiscretisedDensity<3,float>* 
Reconstruction::
construct_target_image_ptr() const
{
  return
      new VoxelsOnCartesianGrid<float> (*get_parameters().proj_data_ptr->get_proj_data_info_ptr(),
					get_parameters().zoom,
					CartesianCoordinate3D<float>(get_parameters().Zoffset,
								     get_parameters().Yoffset,
								     get_parameters().Xoffset),
					CartesianCoordinate3D<int>(get_parameters().output_image_size_z,
                                                                   get_parameters().output_image_size_xy,
                                                                   get_parameters().output_image_size_xy)
                                       );
}



Succeeded 
Reconstruction::
reconstruct() 
{
  shared_ptr<DiscretisedDensity<3,float> > target_image_ptr =
    construct_target_image_ptr();
  Succeeded success = reconstruct(target_image_ptr);
  if (success == Succeeded::yes)
  {
    write_basic_interfile(get_parameters().output_filename_prefix, *target_image_ptr);
    
  }
  return success;
}

END_NAMESPACE_STIR
