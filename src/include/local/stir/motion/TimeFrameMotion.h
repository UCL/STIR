//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    Internal GE use only
*/
#ifndef __stir_motion_TimeFrameMotion__H__
#define __stir_motion_TimeFrameMotion__H__
/*!
  \file
  \ingroup motion
  \brief stir::TimeFrameMotion

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/AbsTimeInterval.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/ParsingObject.h"
#include <time.h> // for localtime

START_NAMESPACE_STIR
/***********************************************************/     
/*! \ingroup motion
  \brief A class for encoding average motion in the frames.


  \par Example par file
  \verbatim
  MoveImage Parameters:=
  input file:= input_filename
  ; output name
  ; filenames will be constructed by appending _f#g1d0b0 (and the extension)
  ; where # is the frame number
  output filename prefix:= output

  ; see TimeFrameDefinitions
  time frame_definition filename := frame_definition_filename
  
  ; next parameter is optional (and not normally necessary)
  ; it can be used if the frame definitions are relative to another scan as what 
  ; is used to for the rigid object motion (i.e. currently the list mode data used
  ;  for the Polaris synchronisation)
  ; scan_start_time_secs_since_1970_UTC

  ; next parameter defines transformation 'direction', defaults to 1
  ;move_to_reference := 1

  ; next can be set to do only 1 frame, defaults means all frames
  ;frame_num_to_process := -1

  ; specify motion, see stir::RigidObject3DMotion
  Rigid Object 3D Motion Type := type

  ; specify reference position, see stir::AbsTimeInterval
  time interval for reference position type:= type

  ; Change output file format, defaults to Interfile. See OutputFileFormat.
  ;Output file format := interfile
  END :=
\endverbatim
*/  
class TimeFrameMotion : public ParsingObject
{
public:
  /* don't have this constructor, as it would need to be repeated by
     all derived classes anyway.
     Call parse() instead.
  TimeFrameMotion(const char * const par_filename);
  */

  //virtual void process_data();

  void move_to_reference(const bool);
  void set_frame_num_to_process(const int);

  int get_frame_num_to_process() const;
  const RigidObject3DTransformation& 
    get_current_rigid_object_transformation() const;

  const TimeFrameDefinitions&
    get_time_frame_defs() const;

protected:

  TimeFrameDefinitions frame_defs;

  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  string frame_definition_filename;

  double scan_start_time;

  bool do_move_to_reference;

  int frame_num_to_process;     
private:
  shared_ptr<RigidObject3DMotion> ro3d_ptr;
  shared_ptr<AbsTimeInterval> _reference_abs_time_sptr;
  RigidObject3DTransformation _current_rigid_object_transformation;
  
  RigidObject3DTransformation _transformation_to_reference_position;

  int scan_start_time_secs_since_1970_UTC;
};

END_NAMESPACE_STIR
#endif
