//
// $Id$
//
 
#ifndef __stir_recon_buildblock_ReconstructionParameters_h__
#define __stir_recon_buildblock_ReconstructionParameters_h__

/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the ReconstructionParameters class

  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/ParsingObject.h"
#include "stir/shared_ptr.h"
#include "stir/ProjData.h"
#include "stir/IO/OutputFileFormat.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

/*!
  \brief base class for all external reconstruction parameter objects
  \ingroup recon_buildblock
*/
class ReconstructionParameters : public ParsingObject
{
public:

  //! the input projection data file name
  string input_filename;

  //! file name for output reconstructed images
  string output_filename_prefix; 

  //! the output image size in x and y direction
  /*! convention: if -1, use get_num_tangential_poss() of the projection data
  */
  int output_image_size_xy; // KT 10122001 appended _xy

  //! the output image size in x and y direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  int output_image_size_z; // KT 10122001 new

  // KT 20/06/2001 disabled
#if 0
  //! number of views to add (i.e. mashing)
  int num_views_to_add;
#endif
  //! the zoom factor
  double zoom;

  //! offset in the x-direction
  double Xoffset;

  //! offset in the y-direction
  double Yoffset;

  // KT 20/06/2001 new
  //! offset in the z-direction
  double Zoffset;

  //! the maximum absolute ring difference number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int max_segment_num_to_process;

  //! constructor
  ReconstructionParameters();

  //! destructor
  virtual ~ReconstructionParameters() {}


  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();


  //! points to the object for the total input projection data
  shared_ptr<ProjData> proj_data_ptr;


  //! defines the format of the output files
  shared_ptr<OutputFileFormat> output_file_format_ptr; // KT 22/05/2003 new

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
  
  virtual void set_defaults();
  virtual void initialise_keymap();

};
 

END_NAMESPACE_STIR









#endif // __ReconstructionParameters_h__

