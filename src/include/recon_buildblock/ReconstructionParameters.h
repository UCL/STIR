//
// $Id$
//
 
#ifndef __ReconstructionParameters_h__
#define __ReconstructionParameters_h__

/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the ReconstructionParameters class

  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/


#include "KeyParser.h"
#include "ImageFilter.h"
#include "utilities.h"

START_NAMESPACE_TOMO

/*!
 \brief base class for all external reconstruction parameter objects


*/


class ReconstructionParameters : public KeyParser
{
public:

  //! the input projection data file name
  string input_filename;

  //! file name for output reconstructed images
  string output_filename_prefix; // KT 160899 changed name

  //! signals whether or not to display final result
  int disp;

  //! the output image size
  int output_image_size; 

  //! signals whether or not to save intermediate files
  int save_intermediate_files;

  //! number of views to add (i.e. mashing)
  int num_views_to_add;

  //! the zoom factor
  double zoom;

  //! offset in the x-direction
  double Xoffset;

  //! offset in the y-direction
  double Yoffset;

  //! the maximum absolute ring difference number to use in the reconstruction
  int max_segment_num_to_process;


  // KT&CL 160899 add arguments
  /* conventions
     output_image_size : if -1, use num_bins
     max_segment_num_to_process : convention: if -1, use get_max_segment()
  */

  //! constructor
  ReconstructionParameters();
         
   
   // has to be called by derived classes to show current parameters
   //CL 01/98/99 return string  
  // KT 160899 added const to method

  //! lists the parameter values
  virtual string parameter_info() const;


  //MJ 03/02/2000 added

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();


  //! points to the object for the total input projection data
  PETSinogramOfVolume *proj_data_ptr;

      




protected:
 
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();
 

};
 

END_NAMESPACE_TOMO









#endif // __ReconstructionParameters_h__

