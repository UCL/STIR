//
// $Id$
//

/*! 
  \file 
  \brief FBP3DRP Reconstruction Parameters implementation.
  \author Claire LABBE
  \author Kris Thielemans
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/FBP3DRP/FBP3DRPParameters.h"
#include "stir/utilities.h"

#ifndef STIR_NO_NAMESPACE
using std::ends;
using std::cout;
using std::endl;
#endif
START_NAMESPACE_STIR



FBP3DRPParameters::FBP3DRPParameters(const string& parameter_filename)

        : ReconstructionParameters()
{
  initialise(parameter_filename);
}

void 
FBP3DRPParameters::
set_defaults()
{
  ReconstructionParameters::set_defaults();

  alpha_colsher_axial = 1;
  fc_colsher_axial = 0.5;
  alpha_colsher_planar = 1;
  fc_colsher_planar = 0.5;
  alpha_ramp = 1;
  fc_ramp = 0.5;
  
  num_segments_to_combine = -1;

  PadS = 2;
  PadZ = 2;
    
  disp=0;
  save_intermediate_files=0;
 
}

void 
FBP3DRPParameters::initialise_keymap()
{
  ReconstructionParameters::initialise_keymap();

  parser.add_start_key("FBP3DRPParameters");
  parser.add_stop_key("End");

  // parser.add_key("Read data into memory all at once",
  //    &on_disk );
  parser.add_key("image to be used for reprojection", &image_for_reprojection_filename);
  parser.add_key("Save intermediate images", &save_intermediate_files);

  // TODO move to 2D recon
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
  parser.add_key("Alpha parameter for Ramp filter",  &alpha_ramp);
  parser.add_key("Cut-off for Ramp filter (in cycles)",&fc_ramp);
  
  parser.add_key("Transaxial extension for FFT", &PadS);
  parser.add_key("Axial extension for FFT", &PadZ);
  
  parser.add_key("Alpha parameter for Colsher filter in axial direction", 
     &alpha_colsher_axial);
  parser.add_key("Cut-off for Colsher filter in axial direction (in cycles)",
    &fc_colsher_axial);
  parser.add_key("Alpha parameter for Colsher filter in planar direction",
    &alpha_colsher_planar);
  parser.add_key("Cut-off for Colsher filter in planar direction (in cycles)",
    &fc_colsher_planar);

}



void 
FBP3DRPParameters::ask_parameters()
{ 
   
  ReconstructionParameters::ask_parameters();
    
   // bool on_disk =  !ask("(1) Read data into memory all at once ?", false);
// TODO move to ReconstructionParameters

    
// PARAMETERS => DISP
    
    disp = ask_num("Which images would you like to display \n\t(0: None, 1: Final, 2: intermediate, 3: after each view) ? ", 0,3,0);

    save_intermediate_files =ask_num("Would you like to save all the intermediate images ? ",0,1,0 );

    image_for_reprojection_filename =
      ask_string("filename of image to be reprojected (empty for using FBP)");

// PARAMETERS => ZEROES-PADDING IN FFT (PADS, PADZ)
    cout << "\nFilter parameters for 2D and 3D reconstruction";
    PadS = ask_num("  Transaxial extension for FFT : ",0,2, 2); 
    PadZ = ask_num(" Axial extension for FFT :",0,2, 2);

// PARAMETERS => 2D RECONSTRUCTION RAMP FILTER (ALPHA, FC)
    cout << endl << "For 2D reconstruction filtering (Ramp filter) : " ;

    num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)",-1,101,-1);
    // TODO check odd
    alpha_ramp =  ask_num(" Alpha parameter for Ramp filter ? ",0.,1., 1.);
    
   fc_ramp =  ask_num(" Cut-off frequency for Ramp filter ? ",0.,.5, 0.5);

// PARAMETERS => 3D RECONSTRUCTION COLSHER FILTER (ALPHA, FC)
    cout << "\nFor 3D reconstruction filtering  (Colsher filter) : ";
    
    alpha_colsher_axial =  ask_num(" Alpha parameter for Colsher filter in axial direction ? ",0.,1., 1.);
    
    fc_colsher_axial =  ask_num(" Cut-off frequency fo Colsher filter in axial direction ? ",0.,.5, 0.5);

    
    alpha_colsher_planar =  ask_num(" Alpha parameter for Colsher filter in planar direction ? ",0.,1., 1.);
    
    fc_colsher_planar =  ask_num(" Cut-off frequency fo Colsher filter in planar direction ? ",0.,.5, 0.5);



}

END_NAMESPACE_STIR
