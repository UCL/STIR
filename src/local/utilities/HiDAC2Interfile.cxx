//
// $Id$
//

/*! 
  \brief Convertions from HiDAC to Interfile images
  
  \author Sanida Mustafovic
   */ 
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/QHidac/HIDAC2Interfile.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif





USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{

  if (argc!=2)
  {
    cerr<<endl<<"Usage: HiDACInterfile <file name>\n"
	<< "Filename is (*.prj/*.i3d)"<<endl<<endl;
    return EXIT_FAILURE;
  }

 string str = argv [1];
 int pos = str.find(".prj");

 if (pos!=string::npos)
  {
    write_interfile_header_for_HiDAC_sinogram(argv[1]);
    
  }
  else
  {
   write_interfile_header_for_HiDAC_image(argv[1]);
  }

   return 
      EXIT_SUCCESS;
 
}



