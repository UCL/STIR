//
// $Id$
//
/*!

  \file
  \ingroup OSMAPOSL
  \brief main() for OSMAPOSL

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "OSMAPOSL/OSMAPOSLReconstruction.h"
#include "VoxelsOnCartesianGrid.h"
#include "shared_ptr.h"


USING_NAMESPACE_TOMO

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  OSMAPOSLReconstruction reconstruction_object(argc>1?argv[1]:"");
  

  shared_ptr<DiscretisedDensity<3, float> > target_image_ptr;     

  // TODO somehow get rid of VoxelsOnCartesianGrid
  if(reconstruction_object.get_parameters().initial_image_filename=="1")
  {
    target_image_ptr =
      new VoxelsOnCartesianGrid<float> (*reconstruction_object.get_parameters().proj_data_ptr->get_proj_data_info_ptr());
    target_image_ptr->fill(1.0);
  }
  else
    {
      target_image_ptr = 
        DiscretisedDensity<3,float>::read_from_file(reconstruction_object.get_parameters().initial_image_filename);
    }


  reconstruction_object.reconstruct(target_image_ptr);


  return EXIT_SUCCESS;

}

