//
// $Id$
//

/*! 
\file
\ingroup utilities
\brief Conversion from interfile to ECAT 6 cti (image and sinogram data)
\author Kris Thielemans
\author PARAPET project
$Date$
$Revision$

*/


#include "DiscretisedDensity.h"
#include "ProjData.h"
#include "shared_ptr.h"
#include "utilities.h"
#include "CTI/Tomo_cti.h"
#include "CTI/cti_utils.h"
#include <iostream>
#include <vector>
#include <string>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::vector;
using std::string;
#endif

USING_NAMESPACE_TOMO




int main(int argc, char *argv[])
{
  char cti_name[1000], scanner_name[1000] = "";
  vector<string> filenames;
  bool its_an_image = true;
  
  if(argc>=4)
  {
    if (strcmp(argv[1],"-s")==0)
      {
	its_an_image = false;
        strcpy(cti_name,argv[2]);
        int num_files = argc-3;
        argv+=3;
        filenames.reserve(num_files);
        for (; num_files>0; --num_files, ++argv)
          filenames.push_back(*argv);	
      }
    else 
      {
	its_an_image = true;
	strcpy(cti_name,argv[1]);	
        int num_files = argc-3;
        argv+=2;
        filenames.reserve(num_files);
        for (; num_files>0; --num_files, ++argv)
          filenames.push_back(*argv);	
	strcpy(scanner_name,*argv);
      }
  }  
  else 
  {
    cerr<<"\nConversion from data to ECAT6 CTI.\n"
        <<"Usage: 2 possible forms depending on data type\n"
	<< "For sinogram data:\n"
	<< "\tconv_ecat6 -s ECAT6_name1 [ECAT6_name2 ...] orig_filename\n"
	<< "For image data:\n"
	<< "\tconv_ecat6 ECAT6_name1  [ECAT6_name2 ...] orig_name scanner_name\n"
	<< "scanner_name has to be recognised by the Scanner class\n"
	<< "Examples are : \"ECAT 953\", \"RPT\" etc.\n"
	<< "(the quotes are required when used as a command line argument)\n\n"
	<< "I will now ask you the same info interactively...\n\n";
    
    its_an_image = ask("Converting images?",true);
    int num_files = ask_num("Number of files",1,10000,1);
    filenames.reserve(num_files);
    char cur_name[max_filename_length];
    for (; num_files>0; --num_files)
    {
      ask_filename_with_extension(cur_name,"Name of the input file? ",its_an_image?".hv":".hs");
      filenames.push_back(cur_name);
    }
    
    ask_filename_with_extension(cti_name,"Name of the ECAT6 file? ",
      its_an_image ? ".img" : ".scn");
  }

  if (its_an_image)
  {

    shared_ptr<Scanner> scanner_ptr = 
      strlen(scanner_name)==0 ?
      Scanner::ask_parameters() :
      Scanner::get_scanner_from_name(scanner_name);

    // read first image
    cerr << "Reading " << filenames[0] << endl;
    shared_ptr<DiscretisedDensity<3,float> > density_ptr =
      DiscretisedDensity<3,float>::read_from_file(filenames[0]);
  
    Main_header mhead;
    make_ECAT6_main_header(mhead, *scanner_ptr, filenames[0], *density_ptr);
    mhead.num_frames = filenames.size();

    FILE *fptr= cti_create (cti_name, &mhead);
    if (fptr == NULL)
    {
      warning("conv_to_ecat6: error opening output file %s\n", cti_name);
      return EXIT_FAILURE;
    }
    unsigned int frame_num = 1;

    while (1)
    {
      if (DiscretisedDensity_to_ECAT6(fptr,
                                      *density_ptr, 
                                      mhead,
                                      frame_num)
                                      == Succeeded::no)
      {
        fclose(fptr);
        return EXIT_FAILURE;
      }
      if (++frame_num > filenames.size())
      {
        fclose(fptr);
        return EXIT_SUCCESS;
      }
      cerr << "Reading " << filenames[frame_num-1] << endl;
      density_ptr =
        DiscretisedDensity<3,float>::read_from_file(filenames[frame_num-1]);
    }
  }
  else 
  {
 
    // read first data set
    cerr << "Reading " << filenames[0] << endl;
    shared_ptr<ProjData > proj_data_ptr =
      ProjData::read_from_file(filenames[0]);
  
    Main_header mhead;
    make_ECAT6_main_header(mhead, filenames[0], *proj_data_ptr->get_proj_data_info_ptr());
    mhead.num_frames = filenames.size();

    FILE *fptr= cti_create (cti_name, &mhead);
    if (fptr == NULL)
    {
      warning("conv_to_ecat6: error opening output file %s\n", cti_name);
      return EXIT_FAILURE;
    }

    unsigned int frame_num = 1;

    while (1)
    {
      if (ProjData_to_ECAT6(fptr,
                            *proj_data_ptr, 
                            mhead,
                            frame_num)
                            == Succeeded::no)
      {
        fclose(fptr);
        return EXIT_FAILURE;
      }
      if (++frame_num > filenames.size())
      {
        fclose(fptr);
        return EXIT_SUCCESS;
      }
      cerr << "Reading " << filenames[frame_num-1] << endl;
      proj_data_ptr =
        ProjData::read_from_file(filenames[frame_num-1]);
    }
  }  
}


