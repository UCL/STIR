// $Id:
//
/*!

  \file
  \ingroup OSSPS
  \brief main() for OSSPS

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date: 
  $Revision: 
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/OSSPS/OSSPSReconstruction.h"
#include "stir/Succeeded.h"


USING_NAMESPACE_STIR

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  OSSPSReconstruction reconstruction_object(argc>1?argv[1]:"");
  

  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;


}

