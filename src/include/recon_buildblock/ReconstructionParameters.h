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
#include "shared_ptr.h"
#include "ProjData.h"
#include <string>

#ifndef TOMO_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_TOMO

/*!
  \brief base class for all external reconstruction parameter objects
  \ingroup recon_buildblock
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
  /*! convention: if -1, use get_num_tangential_poss()
  */
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
  /*! convention: if -1, use get_max_segment_num()*/
  int max_segment_num_to_process;

  //! constructor
  ReconstructionParameters();

  //! destructor
  virtual ~ReconstructionParameters() {}
   
  //! lists the parameter values
  /*! has to be called by derived classes to show current parameters*/
  virtual string parameter_info() const;


  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();


  //! points to the object for the total input projection data
  shared_ptr<ProjData> proj_data_ptr;


protected:
 
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();
 

  /*! 
  \brief 
  This function initialises all parameters, either via parsing, 
  or by calling ask_parameters() (when parameter_filename is the empty string).

  It should be called in the constructor of the last class in the 
  Parameter hierarchy. At that time, all Interfile keys will have been
  initialised, and ask_parameters() will be the appropriate virtual
  function, such that questions are asked for all parameters.
  */
  void initialise(const string& parameter_filename);

};
 

END_NAMESPACE_TOMO









#endif // __ReconstructionParameters_h__

