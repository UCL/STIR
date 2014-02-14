//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details.
*/
#ifndef __stir_recon_buildblock_ProjDataRebinning_H__
#define __stir_recon_buildblock_ProjDataRebinning_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the stir::ProjDataRebinning class

  \author Kris Thielemans

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
  \brief base class for all rebinning algorithms for 3D PET data
  \ingroup recon_buildblock

  TODO describe what rebinning is and why you need it

  \par Usage

  The utility rebin_projdata provides the user interface to this class.
  What follows is for developers.

  Parameters need to be initialised somehow. This is usually done using the 
  parse() member functions (see ParsingObject). 
  For some parameters, <code>set_some_parameter()</code>
  methods are provided.

  Normal usage is for example as follows
  \code
    shared_ptr<ProjDataRebinning> rebinning_sptr;
    // set parameters somehow
    if (rebinning_sptr->set_up()!= Succeeded::yes)
      error("Disaster. Presumably data are inconsistent")
    if (rebinning_sptr->rebin()!= Succeeded::yes)
      error("Disaster. Presumably run-time error or so");
  \endcode

  \par Info for developers

  For convenience, the class is derived from TimedObject. It is the 
  responsibility of the derived class to run these timers though.

  \todo there should be a method to rebin the data without writing the result to disk
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
    At the end of the rebinning, the final 2D projection data are saved to file as given in
    output_filename_prefix.
    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    rebin()=0;
    
  /*! \name get/set the number of segments to process

      \see max_segment_num_to_process
  */
  //@{
  void set_max_segment_num_to_process(int ns);
  int get_max_segment_num_to_process() const;
  //@}
  /*! get/set file name for output projdata (should be without extension)
   */
  //@{
  void set_output_filename_prefix(const string& s);
  string get_output_filename_prefix() const;
  //@}

  //! set projection data that will be rebinned
  /*! Usually this will be set via a call to parse() */
  void set_input_proj_data_sptr(const shared_ptr<ProjData>&);
  // KTTODO I'm not enthousiastic about the next function
  // why would you need this?
  shared_ptr<ProjData> get_proj_data_sptr();
  
  // TODO needed at all?
  // KTTODO: yes, and change parameters
  //! operations prior to the rebinning
  virtual Succeeded set_up();

  // parameters
 protected:

  //! file name for output projdata (should be without extension)
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

