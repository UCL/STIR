
//
//
// $Id$
//
/*!

  \file
  \ingroup LogLikBased_buildblock
  
  \brief  implementation of the LogLikelihoodBasedAlgorithmParameters class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
        
  \version $Revision$
*/

#include "LogLikBased/LogLikelihoodBasedAlgorithmParameters.h"
#include "utilities.h"
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::endl;
using std::ends;
#endif

START_NAMESPACE_TOMO

//
//MJ 01/02/2000 added
//For iterative MLE algorithms
//

LogLikelihoodBasedAlgorithmParameters::LogLikelihoodBasedAlgorithmParameters()
  :IterativeReconstructionParameters()
{

 
  sensitivity_image_filename = "1";
  
  additive_projection_data_filename = "0"; // all zeroes by default


  add_key("sensitivity image",
    KeyArgument::ASCII, &sensitivity_image_filename);

  // AZ 04/10/99 added
  add_key("additive sinogram",
    KeyArgument::ASCII, &additive_projection_data_filename);

}

void LogLikelihoodBasedAlgorithmParameters::ask_parameters()
{

  IterativeReconstructionParameters::ask_parameters();

  char additive_projection_data_filename_char[max_filename_length], sensitivity_image_filename_char[max_filename_length];
 
 
  ask_filename_with_extension(sensitivity_image_filename_char,"Enter file name of sensitivity image (1 = 1's): ", "");   
 
  sensitivity_image_filename=sensitivity_image_filename_char;

  // AZ&MJ 04/10/99 added

  ask_filename_with_extension(additive_projection_data_filename_char,"Enter file name of additive sinogram data (0 = 0's): ", "");    

  additive_projection_data_filename=additive_projection_data_filename_char;


}

bool LogLikelihoodBasedAlgorithmParameters::post_processing()
{
  if (IterativeReconstructionParameters::post_processing())
    return true;

 
  if (sensitivity_image_filename.length() == 0)
  { warning("You need to specify a sensitivity image\n"); return true; }
 
  return false;
}

// KT&MJ 230899 changed return-type to string
string LogLikelihoodBasedAlgorithmParameters::parameter_info() const
{
  
  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[10000];
  ostrstream s(str, 10000);

  s << IterativeReconstructionParameters::parameter_info();
    

  s << "sensitivity image := "
    << sensitivity_image_filename << endl;

 // AZ 04/10/99 added

  s << "additive sinogram := "
    << additive_projection_data_filename << endl<<endl;

  s<<ends;
  
  return s.str();
}

END_NAMESPACE_TOMO
