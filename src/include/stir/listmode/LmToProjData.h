#ifndef __stir_listmode_LmToProjData_H__
#define __stir_listmode_LmToProjData_H__
//
// $Id$
//
/*!
  \file 
  \ingroup listmode

  \brief Program to bin listmode data to 3d sinograms
 
  \author Kris Thielemans
  \author Sanida Mustafovic
  
  $Date$
  $Revision $
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/ProjDataInfo.h"
#include "local/stir/listmode/CListModeData.h"
#include "stir/ParsingObject.h"
#include "stir/TimeFrameDefinitions.h"

#include "stir/recon_buildblock/BinNormalisation.h"

START_NAMESPACE_STIR

class CListEvent;
class CListTime;

class LmToProjData : public ParsingObject
{
public:

  LmToProjData(const char * const par_filename);
  
  LmToProjData();

  shared_ptr<CListModeData> lm_data_ptr;
  TimeFrameDefinitions frame_defs;

  virtual void process_data();
  
protected:

  //! stores the time (in secs) recorded in the previous timing event
  double current_time;
  unsigned int current_frame_num;
  
  //! will be called when a new time frame starts
  /*! The frame number starts from 1. */
  virtual void start_new_time_frame(const unsigned int new_frame_num);

  //! will be called after a new timing event is found in the file
  virtual void process_new_time_event(const CListTime&);

  //! will be called to get the bin for a coincidence event
  /*! If bin.get_bin_value()<=0, the event will be ignored. Otherwise,
    the value will be used as a bin-normalisation factor. */
  virtual void get_bin_from_event(Bin& bin, const CListEvent&) const;

  int get_compression_count(const Bin& bin) const;
  //virtual 
  void do_post_normalisation(Bin& bin) const;

  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  string input_filename;
  string output_filename_prefix;
  string template_proj_data_name;
  string frame_definition_filename;
  bool do_pre_normalisation;
  bool store_prompts;
  int delayed_increment;
  int num_segments_in_memory;
  // TODO make long (or even unsigned long) but can't do this yet because we can't parse longs yet
  int num_events_to_store;
  int max_segment_num_to_process;

  bool interactive;

  shared_ptr<ProjDataInfo> template_proj_data_info_ptr;
  shared_ptr<BinNormalisation> normalisation_ptr;
  shared_ptr<BinNormalisation> post_normalisation_ptr;
  shared_ptr<ProjDataInfo> proj_data_info_cyl_uncompressed_ptr;
  shared_ptr<Scanner> scanner_ptr;
  

  bool do_time_frame;
};

END_NAMESPACE_STIR

#endif
