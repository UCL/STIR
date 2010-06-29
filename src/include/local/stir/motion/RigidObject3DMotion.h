//
// $Id$
//
/*
    Copyright (C) 2003- $Date$ , Hammersmith Imanet Ltd
    For internal GE use only
*/
#ifndef __stir_motion_RigidObject3DMotion__H__
#define __stir_motion_RigidObject3DMotion__H__
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::RigidObject3DMotion

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$
*/

#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include <vector>

START_NAMESPACE_STIR

class AbsTimeInterval;

/*! \ingroup motion

  \brief Base class for 3D rigid motion

  This is really a class for encoding motion of an object in a scanner. So, there is 
  some stuff in here to go from tracker coordinates to scanner coordinates etc.

  Preliminary. Things that need to be worked out:

  - time issues. Relative time is supposed to be relative to the scan start, but 
    this is really dependent on the derived class. It would be far 
    better to stick to secs_since_1970 in the class hierarchy, and use have a "set_reference_time" 
    member here or so.

  - synchronisation: this is supposed to synchornise the tracker clock to a master clock. Again, that behaviour 
    is completely dependent on what the derived class does.
    
*/
class RigidObject3DMotion: public RegisteredObject<RigidObject3DMotion>,
                           public ParsingObject
{

public:
  virtual ~RigidObject3DMotion() {}

  //! get motion in tracker coordinates  
  virtual
    RigidObject3DTransformation 
    get_motion_in_tracker_coords_rel_time(const double time) const =0;

  //! get motion in scanner coordinates
  virtual 
    RigidObject3DTransformation 
    get_motion_in_scanner_coords_rel_time(const double time) const;

  //! \name Average motion for a time interval
  //@{
  virtual 
  RigidObject3DTransformation
  compute_average_motion_in_tracker_coords(const AbsTimeInterval&) const;

  virtual 
  RigidObject3DTransformation
  compute_average_motion_in_scanner_coords(const AbsTimeInterval&) const;

  virtual RigidObject3DTransformation 
    compute_average_motion_in_tracker_coords_rel_time(const double start_time, const double end_time)const = 0;

  virtual RigidObject3DTransformation 
    compute_average_motion_in_scanner_coords_rel_time(const double start_time, const double end_time)const;

  //@}

  //! Info on when the motion was determined
  /*! Will return a vector of doubles filled with the sampling times between
      \a start_time and \a end_time.

      \todo Really only makes sense for motion tracking that happens via sampling.
      One could imagine having simulated motion, and then this function wouldn't make
      a lot of sense. So, it probably should be moved to a derived class
      \c SampledRigidObject3DMotion or so.
  */
  virtual std::vector<double>
    get_rel_time_of_samples(const double start_time, const double end_time)const = 0;

  //! Has to be called and will be used to synchronise the target-system time and motion tracking time
  /*! In practice, this should make sure that a 'rel_time' of 0 corresponds to the start of the scan
   */
  virtual Succeeded synchronise() =0;


  virtual double secs_since_1970_to_rel_time(std::time_t) const = 0;


 protected:
#if 0
  //!  Option to set  time offset manually in case synchronisation cannot be performed
  void
    set_time_offset(const double time_offset);
  double 
    get_time_offset() const;
#endif
  //! Temporary (?) function to allow base class to see if synchronised was called or not
  virtual bool is_synchronised() const = 0;
 public:

  virtual const RigidObject3DTransformation& 
    get_transformation_to_scanner_coords() const = 0;
  virtual const RigidObject3DTransformation& 
    get_transformation_from_scanner_coords() const = 0;

  virtual void  
    set_transformation_from_scanner_coords(const RigidObject3DTransformation&) = 0;

protected:
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

END_NAMESPACE_STIR

#endif
