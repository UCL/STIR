//
// $Id$: $Date$
//

/*! 
  \file
  \ingroup utilities
  \brief Conversion from ECAT 6 cti to interfile (image and sinogram data)
  \author Kris Thielemans
  \author Damien Sauge
  \author Sanida Mustafovic
  \author PARAPET project
  \version $Revision$
  \date $Date$

  \warning Most of the data in the ECAT 6 headers is ignored (except dimensions)
  \warning Data are scaled using the subheader.scale_factor * subheader.loss_correction_fctr
  (unless the loss correction factor is < 0, in which case it is assumed to be 1).
  \warning Currently, the decay correction factor is ignored.
*/


#include "utilities.h"
#include "interfile.h"
#include "shared_ptr.h"
#include "VoxelsOnCartesianGrid.h"
#include "CTI/Tomo_cti.h"
#include "CTI/cti_utils.h"
#include <stdio.h>
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif



USING_NAMESPACE_TOMO

int
main(int argc, char *argv[])
{
    char cti_name[max_filename_length], out_name[max_filename_length];
    FILE *cti_fptr;
 
    if(argc==3)
      { 
	strcpy(cti_name, argv[2]);
	strcpy(out_name, argv[1]);
      }
    else 
    {
        cerr<<"\nConversion from ECAT6 CTI data to interfile.\n";
        cerr<<"Usage: convecat6_if [output_file_name cti_data_file_name]\n"<<endl;
	if (argc!=1)
	  exit(EXIT_FAILURE);

	ask_filename_with_extension(out_name,"Name of the output file? (.hv/.hs and .v/.s will be added)","");    
        ask_filename_with_extension(cti_name,"Name of the input data file? ",".scn");
        
    }


// open input file, read main header
    cti_fptr=fopen(cti_name, "rb"); 
    if(!cti_fptr) {
        error("\nError opening input file: %s\n",cti_name);
    }
    Main_header mhead;
    if(cti_read_main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) {
        error("\nUnable to read main header in file: %s\n",cti_name);
    }

    const int frame_num=
      ask_num("Frame number ? ",
             1,
#ifndef TOMO_NO_NAMESPACES
             // VC needs this
             std::
#endif
             max(static_cast<int>(mhead.num_frames),1),
             1);
    const int bed_num=
      ask_num("Bed number ? ",
              0,static_cast<int>(mhead.num_bed_pos), 0);
    const int gate_num=
      ask_num("Gate number ? ",
              1,
#ifndef TOMO_NO_NAMESPACES
             // VC needs this
             std::
#endif
              max(static_cast<int>(mhead.num_gates),1), 
              1);
    const int data_num=
      ask_num("Data number ? ",0,8, 0);

        
    switch(mhead.file_type)
    { 
    case matImageFile:
      {                       
        shared_ptr<VoxelsOnCartesianGrid<float> > image_ptr =
          ECAT6_to_VoxelsOnCartesianGrid(frame_num, gate_num, data_num, bed_num,
                         cti_fptr, mhead);
        write_basic_interfile(out_name,*image_ptr);
        break;
      }
    case matScanFile:
    case matAttenFile:
      {            
        const int max_ring_diff= 
           ask_num("Max ring diff to store (-1 == num_rings-1)",-1,100,-1);
        
        ECAT6_to_PDFS(frame_num, gate_num, data_num, bed_num,
                         max_ring_diff, 
                         out_name, cti_fptr, mhead);
        break;
      }
    default:
      {
        error("\nSupporting only image, scan or atten file type at the moment. Sorry.\n");            
      }
    }    
    fclose(cti_fptr);
    
    return EXIT_SUCCESS;
}
