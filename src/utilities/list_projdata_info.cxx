/*!
  \file
  \ingroup utilities

  \brief A utility that lists size info on the projection data on stdout.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"

#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
#endif


USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{ 
  
  if(argc!=2) 
  {
    cerr<<"Usage: " << argv[0] << " projdata_file\n";
    return EXIT_FAILURE;

  }

  shared_ptr<ProjData> projdata_ptr = 
  ProjData::read_from_file(argv[1], ios::in|ios::out);

  cout << "Info for file " << argv[1] << '\n';
  cout << projdata_ptr->get_proj_data_info_ptr()->parameter_info() << endl;

  return EXIT_SUCCESS;
}
