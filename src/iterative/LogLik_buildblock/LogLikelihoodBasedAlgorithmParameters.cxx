//
// $Id$
//
/*!

  \file
  \ingroup LogLikBased_buildblock
  
  \brief  implementation of the LogLikelihoodBasedAlgorithmParameters class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic  
  \author PARAPET project
      
  $Date$      
  $Revision$
*/

#include "LogLikBased/LogLikelihoodBasedAlgorithmParameters.h"
#include "utilities.h"
#include <iostream>
#ifndef USE_PMRT
#include "recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#else
#include "recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
#include "recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"
#ifndef TOMO_NO_NAMESPACES
using std::endl;
using std::ends;
#endif

START_NAMESPACE_TOMO


LogLikelihoodBasedAlgorithmParameters::LogLikelihoodBasedAlgorithmParameters()
:IterativeReconstructionParameters()
{
}

void
LogLikelihoodBasedAlgorithmParameters::set_defaults()
{
  IterativeReconstructionParameters::set_defaults();

  sensitivity_image_filename = "1";  
  additive_projection_data_filename = "0"; // all zeroes by default


  // set default for projector_pair_ptr
#ifndef USE_PMRT
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr =
    new ForwardProjectorByBinUsingRayTracing();
  shared_ptr<BackProjectorByBin> back_projector_ptr =
    new BackProjectorByBinUsingInterpolation();
#else
  shared_ptr<ProjMatrixByBin> PM = 
    new  ProjMatrixByBinUsingRayTracing(); 	
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr =
    new ForwardProjectorByBinUsingProjMatrixByBin(PM); 
  shared_ptr<BackProjectorByBin> back_projector_ptr =
    new BackProjectorByBinUsingProjMatrixByBin(PM); 
#endif

  projector_pair_ptr = 
    new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr);
}

void
LogLikelihoodBasedAlgorithmParameters::initialise_keymap()
{

  IterativeReconstructionParameters::initialise_keymap();
  parser.add_parsing_key("Projector pair type", &projector_pair_ptr);

  parser.add_key("sensitivity image", &sensitivity_image_filename);

  parser.add_key("additive sinogram",&additive_projection_data_filename);

}
void LogLikelihoodBasedAlgorithmParameters::ask_parameters()
{

  IterativeReconstructionParameters::ask_parameters();

  projector_pair_ptr = ProjectorByBinPair::ask_type_and_parameters();

  char additive_projection_data_filename_char[max_filename_length], sensitivity_image_filename_char[max_filename_length];
 
  ask_filename_with_extension(sensitivity_image_filename_char,"Enter file name of sensitivity image (1 = 1's): ", "");   
 
  sensitivity_image_filename=sensitivity_image_filename_char;

  ask_filename_with_extension(additive_projection_data_filename_char,"Enter file name of additive sinogram data (0 = 0's): ", "");    

  additive_projection_data_filename=additive_projection_data_filename_char;


}

bool LogLikelihoodBasedAlgorithmParameters::post_processing()
{
  if (IterativeReconstructionParameters::post_processing())
    return true;

  if (projector_pair_ptr.use_count() == 0)
  { warning("No valid projector pair is defined\n"); return true; }

  if (projector_pair_ptr->get_forward_projector_sptr().use_count() == 0)
  { warning("No valid forward projector is defined\n"); return true; }

  if (projector_pair_ptr->get_back_projector_sptr().use_count() == 0)
  { warning("No valid back projector is defined\n"); return true; }

  if (sensitivity_image_filename.length() == 0)
  { warning("You need to specify a sensitivity image\n"); return true; }
 
  return false;
}



END_NAMESPACE_TOMO
