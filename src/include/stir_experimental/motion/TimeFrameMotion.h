//
//
/*
    Copyright (C) 2003- 2006, Hammersmith Imanet Ltd
    Internal GE use only
*/
#ifndef __stir_motion_TimeFrameMotion__H__
#define __stir_motion_TimeFrameMotion__H__
/*!
  \file
  \ingroup motion
  \brief stir::TimeFrameMotion

  \author Kris Thielemans
*/

#include "stir_experimental/motion/RigidObject3DTransformation.h"
#include "stir_experimental/motion/RigidObject3DMotion.h"
#include "stir_experimental/AbsTimeInterval.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/ParsingObject.h"

START_NAMESPACE_STIR
/***********************************************************/     
/*! \ingroup motion
  \brief A class for encoding average motion in the frames.


  \par Example par file
  These are the parameters set by this class. The derived classes will
  add more.

  \verbatim

  ; see stir::TimeFrameDefinitions
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

  virtual Succeeded process_data() = 0;

  void move_to_reference(const bool);
  void set_frame_num_to_process(const int);

  int get_frame_num_to_process() const;
  //! get transformation from (or to) reference for current frame
  /*! This is computed using 
    RigidObject3DTransformation::compute_average_motion_in_scanner_coords
    for the current frame.
  */
  const RigidObject3DTransformation& 
    get_current_rigid_object_transformation() const;

  //! Get the transformation to the reference as returned by the RigidObject3DMotion object
  const RigidObject3DTransformation& 
    get_rigid_object_transformation_to_reference() const
    { return _transformation_to_reference_position; }

  const TimeFrameDefinitions&
    get_time_frame_defs() const;

  double get_frame_start_time(unsigned frame_num) const
  { return _frame_defs.get_start_time(frame_num) + _scan_start_time; }

  double get_frame_end_time(unsigned frame_num) const
  { return _frame_defs.get_end_time(frame_num) + _scan_start_time; }
  
  const RigidObject3DMotion& get_motion() const
  { return *_ro3d_sptr; }

protected:


  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();


private:
  std::string _frame_definition_filename;
  bool _do_move_to_reference;

  TimeFrameDefinitions _frame_defs;
  shared_ptr<RigidObject3DMotion> _ro3d_sptr;
  shared_ptr<AbsTimeInterval> _reference_abs_time_sptr;
  RigidObject3DTransformation _current_rigid_object_transformation;
  
  RigidObject3DTransformation _transformation_to_reference_position;

  int _scan_start_time_secs_since_1970_UTC;
  double _scan_start_time;

  int _frame_num_to_process;     
};

END_NAMESPACE_STIR
#endif
