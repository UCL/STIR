//
// $Id$
//
#ifndef __stir_recon_buildblock_ProjDataRebinning_H__
#define __stir_recon_buildblock_ProjDataRebinning_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the ProjDataRebinning class

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
/* Modification history
*/


#include "stir/RegisteredObject.h"
#include "stir/TimedObject.h"
#include "stir/ParsingObject.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include <string>


#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR


class Succeeded;

/*!
  \brief base class for all ProjDataRebinnings
  \ingroup recon_buildblock

  For convenience, the class is derived from TimedObject. It is the 
  responsibility of the derived class to run these timers though.

*/


class ProjDataRebinning : 
  public TimedObject, public ParsingObject ,
  public RegisteredObject<ProjDataRebinning >
{
public:
  //! virtual destructor
  virtual ~ProjDataRebinning();
  
  //! gives method information
  virtual string method_info() const = 0;
  
  //! executes the rebinning
  /*!
    Calls 1 argument rebin().
    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    rebin();
  //! executes the ProjDataRebinning
  /*!
    At the end of the rebinning, the final image is saved to file as given in 
    output_filename_prefix. 
    \return Succeeded::yes if everything was alright.
   */     
#if 0
  virtual Succeeded 
    rebin(const ProjData&) = 0;
#endif
  // TODO needed at all?

  //! operations prior to the rebinning
  virtual Succeeded set_up();

  // parameters
 protected:

  //! file name for output projdata (can be without extension)
  string output_filename_prefix; 
  //! file name for input projdata
  string input_filename; 
  //! the maximum absolute segment number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int max_segment_num_to_process;

protected:
#if 0
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
#endif

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();  
  virtual void set_defaults();
  virtual void initialise_keymap();

 protected: // members

  shared_ptr<ProjData> proj_data_sptr;
};

END_NAMESPACE_STIR

    
#endif

