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
#include "local/stir/listmode/TimeFrameDefinitions.h"

#include "stir/recon_buildblock/BinNormalisation.h"

START_NAMESPACE_STIR

class CListEvent;

class LmToProjData : public ParsingObject
{
public:

  LmToProjData(const char * const par_filename);
  
  LmToProjData();

  shared_ptr<CListModeData> lm_data_ptr;
  TimeFrameDefinitions frame_defs;
  virtual void get_bin_from_event(Bin& bin, const CListEvent& record, 
				  const double time) const;

  void compute();
  shared_ptr<BinNormalisation> normalisation_ptr;
  shared_ptr<ProjDataInfo> proj_data_info_cyl_uncompressed_ptr;
  
protected:

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  string input_filename;
  string output_filename_prefix;
  string template_proj_data_name;
  string frame_definition_filename;
  bool pre_or_post_normalisation;
  bool store_prompts;
  int delayed_increment;
  int current_frame;
  int num_segments_in_memory;
  long num_events_to_store;
  int max_segment_num_to_process;

  bool interactive;

  shared_ptr<ProjDataInfo> template_proj_data_info_ptr;

  bool do_time_frame;
};

END_NAMESPACE_STIR

#endif
