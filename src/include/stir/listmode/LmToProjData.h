#ifndef __stir_listmode_LmToProjData_H__
#define __stir_listmode_LmToProjData_H__
//
//
/*!
  \file 
  \ingroup listmode

  \brief Declaration of the stir::LmToProjData class which is used to bin listmode data to (3d) sinograms
 
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Daniel Deidda
  
*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, 2021, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "stir/listmode/LmToProjDataAbstract.h"
#include "stir/ProjDataInfo.h"
#include "stir/listmode/ListModeData.h"
#include "stir/ParsingObject.h"
#include "stir/TimeFrameDefinitions.h"

#include "stir/recon_buildblock/BinNormalisation.h"

START_NAMESPACE_STIR

class ListEvent;
class ListTime;

/*!
  \ingroup listmode

  \brief This class is used to bin listmode data to projection data,
  i.e. (3d) sinograms.

  It provides the basic machinery to go through a list mode data file,
  and write projection data for each time frame. 

  The class can parse its parameters from an input file. This has the
  following format:

  \verbatim
  lm_to_projdata Parameters:=

  input file := some_lm_file
  output filename prefix := my_favorite_name_for_the_projdata

  ; parameters that determine the sizes etc of the output

    template_projdata := some_projdata_file
    ; the next can be used to use a smaller number of segments than given 
    ; in the template
    maximum absolute segment number to process := 

  ; parameters for saying which events will be stored

    ; time frames (see TimeFrameDefinitions doc for format)
    frame_definition file := frames.fdef
    ; or a total number of events (if  larger than 0, frame definitions will be ignored)
    ; note that this normally counts the total of prompts-delayeds (see below)
    num_events_to_store := -1

  ; parameters relating to prompts and delayeds

    ; with the default values, prompts will be added and delayed subtracted
    ; to give the usual estimate of the trues.

    ; store the prompts (value should be 1 or 0)
    store prompts := 1  ;default
    ; what to do if it's a delayed event
    store delayeds := 1  ;default


  ; parameters related to normalisation
  ; default settings mean no normalisation
  ; Use with care!

    ; in pre normalisation, each event will contribute its 
    ; 'normalisation factor' to the bin
    ; in post normalisation, an average factor for the bin will be used
    do pre normalisation  := 0 ; default is 0
    ; type of pre-normalisation (see BinNormalisation doc)
    Bin Normalisation type for pre-normalisation := None ; default
    ; type of post-normalisation (see BinNormalisation doc)
    Bin Normalisation type for post-normalisation := None ; default

  ; miscellaneous parameters

    ; list each event on stdout and do not store any files (use only for testing!)
    ; has to be 0 or 1
    List event coordinates := 0

    ; if you're short of RAM (i.e. a single projdata does not fit into memory),
    ; you can use this to process the list mode data in multiple passes.
    num_segments_in_memory := -1

  End := 
  \endverbatim
  
  Hopefully the only thing that needs explaining are the parameters related
  to prompts and delayeds. These are used to allow different ways of
  processing the data. There are really only 3 useful cases:

  <ul>
  <li> 'online' subtraction of delayeds<br>
       This is the default, and adds prompts but subtracts delayeds.
       \code
    store prompts := 1 
    store delayeds := 1
       \endcode
  </li>
  <li> store prompts only<br>
       Use 
       \code
    store prompts := 1 
    store delayeds := 0
       \endcode

  </li>
  <li> store delayeds only<br>
       Use 
       \code
    store prompts := 0 
    store delayeds := 1
       \endcode
       Note that now the delayted events will be <strong>added</strong>,
       not subtracted.
  </li>
  </ul>

  \par Notes for developers

  The class provides several
  virtual functions. If a derived class overloads these, the default behaviour
  might change. For example, get_bin_from_event() might do motion correction.

  \todo Currently, there is no support for gating or energy windows. This
  could in principle be added by a derived class, but it would be better
  to do it here.
  \todo Timing info or so for get_bin_from_event() for rotating scanners etc.
  \todo There is overlap between the normalisation and the current treatment
  of bin.get_bin_value(). This is really because we should be using 
  something like a EventNormalisation class for pre-normalisation.

  \see ListModeData for more info on list mode data.

*/

class LmToProjData : public LmToProjDataAbstract
{
public:

  //! Constructor taking a filename for a parameter file
  /*! Will attempt to open and parse the file. */
  LmToProjData(const char * const par_filename);

  //! Default constructor
  /*! \warning leaves parameters ill-defined. Set them by parsing. */
  LmToProjData();

  /*! \name Functions to get/set parameters
    This can be used as alternative to the parsing mechanism.
   \warning Be careful with setting shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  void set_template_proj_data_info_sptr(shared_ptr<const ProjDataInfo>);
  shared_ptr<ProjDataInfo> get_template_proj_data_info_sptr();

  //! \brief set input data
  /*! will throw of the input data is not of type \c ListModeData */
  virtual void set_input_data(const shared_ptr<ExamData>&);
  //! \brief set input data
  /*! will throw of the input data is not of type \c ListModeData */
  virtual void set_input_data(const std::string& filename);
#if 0
  //! get input data
  /*! Will throw an exception if it wasn't set first */
  virtual ListModeData& get_input_data() const;
#endif

  void set_output_filename_prefix(const std::string&);
  std::string get_output_filename_prefix() const;

  void set_store_prompts(bool);
  bool get_store_prompts() const;
  void set_store_delayeds(bool);
  bool get_store_delayeds() const;
  void set_num_segments_in_memory(int);
  int get_num_segments_in_memory() const;
  void set_num_events_to_store(long int);
  long int get_num_events_to_store() const;
  void set_time_frame_definitions(const TimeFrameDefinitions&);
  const TimeFrameDefinitions& get_time_frame_definitions() const;
  //@}

  //! Perform various checks
  /*! Note: this is currently called by post_processing(). This will change in version 5.0 */
  virtual Succeeded set_up();

  //! This function does the actual work
  virtual void process_data(shared_ptr<ProjData> proj_data_sptr = nullptr);

protected:
  
  //! will be called when a new time frame starts
  /*! The frame numbers start from 1. */
  virtual void start_new_time_frame(const unsigned int new_frame_num);

  //! will be called after a new timing event is found in the file
  virtual void process_new_time_event(const ListTime&);

  //! will be called to get the bin for a coincidence event
  /*! If bin.get_bin_value()<=0, the event will be ignored. Otherwise,
    the value will be used as a bin-normalisation factor 
    (on top of anything done by normalisation_ptr). 
    \todo Would need timing info or so for e.g. time dependent
    normalisation or angle info for a rotating scanner.*/
  virtual void get_bin_from_event(Bin& bin, const ListEvent&) const;

  //! A function that should return the number of uncompressed bins in the current bin
  /*! \todo it is not compatiable with e.g. HiDAC doesn't belong here anyway
      (more ProjDataInfo?)
  */
  int get_compression_count(const Bin& bin) const;
  
  //! Computes a post-normalisation factor (if any) for this bin
  /*! This uses get_compression_count() when do_pre_normalisation=true, or
      post_normalisation_ptr otherwise. 
  */
  void do_post_normalisation(Bin& bin) const;

  //! \name parsing functions
  //@{
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  //@}

  //! \name parsing variables
  //{@
  std::string input_filename;
  std::string output_filename_prefix;
  std::string template_proj_data_name;
  //! frame definitions
  /*! Will be read using TimeFrameDefinitions */
  std::string frame_definition_filename;
  bool do_pre_normalisation;
  bool store_prompts;
  bool store_delayeds;
  int num_segments_in_memory;
  long int num_events_to_store;
  int max_segment_num_to_process;

  //! Toggle readable output on stdout or actual projdata
  /*! corresponds to key "list event coordinates" */
  bool interactive;

  shared_ptr<ProjDataInfo> template_proj_data_info_ptr;
  //! This will be used for pre-normalisation
  shared_ptr<BinNormalisation> normalisation_ptr;
  //! This will be used for post-normalisation
  shared_ptr<BinNormalisation> post_normalisation_ptr;

  //@}

  //! Pointer to the actual data
  shared_ptr<ListModeData> lm_data_ptr;

  //! Time frames
  TimeFrameDefinitions frame_defs;

  //! stores the time (in secs) recorded in the previous timing event
  double current_time;
  //! stores the current frame number
  /*! The frame numbers start from 1. */
  unsigned int current_frame_num;

  //! Internal variable that will be used for pre-normalisation
  /*! Will be removed when we have EventNormalisation (or similar) hierarchy */
  shared_ptr<const ProjDataInfo> proj_data_info_cyl_uncompressed_ptr;


  /*! \brief variable that will be set according to if we are using 
    time frames or num_events_to_store
  */
  bool do_time_frame;
  //! A variable that will be set to 1,0 or -1, according to store_prompts and store_delayeds
  int delayed_increment;

  //! an internal bool variable to check if the object has been set-up or not
  bool _already_setup;
};

END_NAMESPACE_STIR

#endif
