//
// $Id$
//

/*! 
  \file 
  \brief Class for FBP3DRP Reconstruction Parameters
  \author Claire LABBE
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __FBP3DRPParameters_h__
#define __FBP3DRPParameters_h__

#include "stir/recon_buildblock/ReconstructionParameters.h" 

START_NAMESPACE_STIR

/*!
  \brief Class for FBP3DRP Reconstruction Parameters
*/
class FBP3DRPParameters: public ReconstructionParameters
{
public:
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit FBP3DRPParameters(const string& parameter_filename = "");
   
   
  void FBP3DRPParameters::ask_parameters();

 
public:// FBP3DRP specific parameters. public as FBP3DRPReconstruction class needs access.

  //! Switch to display intermediate images, 0,1,2
  int disp;

  //! Switch to save files after each segment, 0 or 1
  int save_intermediate_files;
   
  //! Filename of image used in the reprojection step (default is empty)
  /*! If the filename is empty, FBP is used (with filter parameters as 
    specified further). 

    \warning This image must have the correct scale. That is, if you use the 
    forward projector on it, you get sinograms of the same scale as the 
    input sinograms. There is NO check on this.
  */  
  string image_for_reprojection_filename;

  //! Transaxial extension for FFT
  int PadS; 
  //! Axial extension for FFT
  int PadZ;
  //! Ramp filter: Alpha value
  double alpha_ramp;
  //! Ramp filter: Cut off frequency
  double fc_ramp;  

  //! Alpha parameter for Colsher filter in axial direction
  double alpha_colsher_axial;
  //! Cut-off frequency for Colsher filter in axial direction
  double fc_colsher_axial;
  //! Alpha parameter for Colsher filter in planar direction
  double alpha_colsher_planar; 
  //! Cut-off frequency for Colsher filter in planar direction
  double fc_colsher_planar; 
 

  //! =1 => apply additional fitting procedure to forward projected data
  int fit_projections; 
private:

  virtual void set_defaults();
  virtual void initialise_keymap();
  
};

END_NAMESPACE_STIR

#endif // __FBP3DRPParameters_h__

