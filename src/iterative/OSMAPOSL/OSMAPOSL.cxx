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

  $Date$
  $Revision$
*/
#include "OSMAPOSL/OSMAPOSLReconstruction.h"
#include "tomo/Succeeded.h"


USING_NAMESPACE_TOMO

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  OSMAPOSLReconstruction reconstruction_object(argc>1?argv[1]:"");
  

  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;


}

