//
// $Id$
//
#ifndef __stir_recon_buildblock_Reconstruction_H__
#define __stir_recon_buildblock_Reconstruction_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the Reconstruction class

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
/* Modification history

   KT 10122001
   - added construct_target_image_ptr and 0 argument reconstruct()
*/


#include "stir/TimedObject.h"
#include "stir/ParsingObject.h"
#include "stir/shared_ptr.h"
#include "stir/ProjData.h"
#include "stir/IO/OutputFileFormat.h"
#include <string>


#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;

/*!
  \brief base class for all Reconstructions
  \ingroup recon_buildblock
  As there is not a lot of commonality between different reconstruction algorithms,
  this base class is rather basic. It essentially takes care of constructing a target
  image, calls the virtual reconstruct() function, and writes the result to file.

  For convenience, the class is derived from TimedObject. It is the 
  responsibility of the derived class to run these timers though.

*/

class Reconstruction : public TimedObject, public ParsingObject 
{
public:
  //! virtual destructor
  virtual ~Reconstruction() {};
  
  //! gives method information
  virtual string method_info() const = 0;
  

  //! Creates a suitable target_image as determined by the parameters
  virtual DiscretisedDensity<3,float>* 
    construct_target_image_ptr() const; // KT 10122001 new
  
  //! executes the reconstruction
  /*!
    Calls construct_target_image_ptr() and then 1 argument reconstruct().
    At the end of the reconstruction, the final image is saved to file as given in 
    ReconstructionParameters::output_filename_prefix. 

    This behaviour can be modified by a derived class (for instance see 
    IterativeReconstruction::reconstruct()).

    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    reconstruct(); // KT 10122001 new

  //! executes the reconstruction
  /*!
    \param target_image_ptr The result of the reconstruction is stored in *target_image_ptr.
    For iterative reconstructions, *target_image_ptr is used as an initial estimate.
    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    reconstruct(shared_ptr<DiscretisedDensity<3,float> > const& target_image_ptr) = 0;

  //! accessor for the external parameters
  Reconstruction& get_parameters()
    {
      return *this;
    }

  //! accessor for the external parameters
  const Reconstruction& get_parameters() const
    {
      return *this;
    }

  // parameters
 protected:

  //! the input projection data file name
  string input_filename;

  //! file name for output reconstructed images
  string output_filename_prefix; 

  //! the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  int output_image_size_xy; // KT 10122001 appended _xy

  //! the output image size in z direction
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


  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();


  //! points to the object for the total input projection data
  shared_ptr<ProjData> proj_data_ptr;

  //! defines the format of the output files
  shared_ptr<OutputFileFormat> output_file_format_ptr; 

protected:
 
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

  /*! 
  \brief 
  This function initialises all parameters, either via parsing, 
  or by calling ask_parameters() (when parameter_filename is the empty string).

  It should be called in the constructor of the last class in the 
  hierarchy. At that time, all Interfile keys will have been
  initialised, and ask_parameters() will be the appropriate virtual
  function, such that questions are asked for all parameters.
  */
  void initialise(const string& parameter_filename);
  
  virtual void set_defaults();
  virtual void initialise_keymap();


};

END_NAMESPACE_STIR

    
#endif

