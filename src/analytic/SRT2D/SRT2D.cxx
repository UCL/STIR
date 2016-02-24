//
// $Id: SRT2D.cxx $
//


#include "stir/analytic/SRT2D/SRT2DReconstruction.h"
#include "stir/Succeeded.h"

USING_NAMESPACE_STIR
    
int main(int argc, char **argv)
{
    SRT2DReconstruction reconstruction_object(argc>1?argv[1]:"");


  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;
} 
  


