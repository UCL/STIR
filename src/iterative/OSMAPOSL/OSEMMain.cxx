//
// $Id$
//

#include "OSEM/OSMAPOSLReconstruction.h"

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  OSMAPOSLReconstruction reconstruction_object(argc>1?argv[1]:"");

  PETImageOfVolume target_image(reconstruction_object.get_parameters().proj_data_ptr->scan_info);

  if(reconstruction_object.get_parameters().initial_image_filename=="1")
    target_image.fill(1.0);
  
  else
    {
      // MJ 05/03/2000 replaced by interfile
      // Note: need absolute replacement of target_image to account for zooming
      target_image = read_interfile_image(reconstruction_object.get_parameters().initial_image_filename.c_str());
    }


  //TODO use warning(),use zoom factor
  if(reconstruction_object.get_parameters().zoom!=1)
    warning("Warning: No zoom factor will be used");


  reconstruction_object.reconstruct(target_image);


  return EXIT_SUCCESS;

}
