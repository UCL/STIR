//
// $Id$: $Date$
//

/*! 
\file
\ingroup utilities
\brief Conversion from interfile to ECAT 6 cti (image and sinogram data)
\author Kris Thielemans
\author PARAPET project
\version $Date$
\date  $Revision$

*/


#include "DiscretisedDensity.h"
#include "ProjData.h"
#include "shared_ptr.h"
#include "utilities.h"
#include "CTI/Tomo_cti.h"

#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_TOMO




int main(int argc, char *argv[])
{
  char header_name[1000], cti_name[1000], scanner_name[1000] = "";
  bool its_an_image = true;
  
  if(argc==4)
  {
    if (strcmp(argv[1],"-s")==0)
      {
	its_an_image = false;
	strcpy(header_name,argv[3]);
	strcpy(cti_name,argv[2]);
      }
    else 
      {
	its_an_image = true;	
	strcpy(header_name,argv[2]);
	strcpy(cti_name,argv[1]);
	strcpy(scanner_name,argv[3]);
      }
  }  
  else 
  {
    cerr<<"\nConversion from data to ECAT6 CTI.\n"
        <<"Usage: 2 possible forms depending on data type\n"
	<< "For sinogram data:\n"
	<< "\tconv_ecat6 -s ECAT6_name orig_name\n"
	<< "For image data:\n"
	<< "\tconv_ecat6 ECAT6_name orig_name scanner_name\n"
	<< "scanner_name has to be recognised by the Scanner class\n"
	<< "Examples are : \"ECAT 953\", \"RPT\" etc.\n"
	<< "(the quotes are required when used as a command line argument)\n\n"
	<< "I will now ask you the same info interactively...\n\n";
    
    ask_filename_with_extension(header_name,"Name of the input file? ",".hs");    
    its_an_image = ask("Is this an image",true);
    ask_filename_with_extension(cti_name,"Name of the ECAT6 file? ",
      its_an_image ? ".img" : ".scn");
  }

  if (its_an_image)
  {

    shared_ptr<Scanner> scanner_ptr = 
      strlen(scanner_name)==0 ?
      Scanner::ask_parameters() :
      Scanner::get_scanner_from_name(scanner_name);

    shared_ptr<DiscretisedDensity<3,float> > density_ptr =
      DiscretisedDensity<3,float>::read_from_file(header_name);

    if (DiscretisedDensity_to_ECAT6(*density_ptr, cti_name, header_name, 
				    *scanner_ptr) == Succeeded::yes)
      return EXIT_SUCCESS;    
    else
      return EXIT_FAILURE;
  }
  else 
  {
    shared_ptr<ProjData> proj_data_ptr=
       ProjData::read_from_file(header_name);

    if (ProjData_to_ECAT6(*proj_data_ptr, cti_name, header_name) == Succeeded::yes)
      return EXIT_SUCCESS;
    else
      return EXIT_FAILURE;
  }
}


