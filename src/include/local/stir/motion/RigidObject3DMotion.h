//
// $Id$
//
/*
    Copyright (C) 2003- $Date$ , Hammersmith Imanet Ltd
    For internal GE use only
*/
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::RigidObject3DMotion

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$
*/

#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/listmode/CListModeData.h"
#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include <ctime>

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; }
#endif


START_NAMESPACE_STIR

/*! \ingroup motion

  \brief Base class for 3D rigid motion

  This is really a class for encoding motion of an object in a scanner. So, there is 
  some stuff in here to go from tracker coordinates to scanner coordinates etc.

  Preliminary. Things that need to be worked out:

  - time issues. Relative time is supposed to be relative to the scan start, but 
    absolute times are currently depending on the derived class. It would be far 
    better to stick to secs_since_1970.

  - synchronisation: currently uses a list mode file

  - motion parameters are w.r.t. a tracker dependend coordinate system. There is then
    some stuff to go to scanner systems etc. But it's laregly left up to the application to 
    do this properly.

  - stuff to get the 'reference position' is in here, but maybe it shouldn't. At present, 
  there are various parameters that can be used to set this. For example, it can use an
  attenuation scan, although that's half-broken (because the ECAT7 header for a .a file 
  does not record the scan duration (as it wouldn't make a lot of sense anyway).
    
*/
class RigidObject3DMotion: public RegisteredObject<RigidObject3DMotion>,
                           public ParsingObject
{

public:
  virtual ~RigidObject3DMotion() {}

  //virtual RigidObject3DTransformation 
  //  compute_average_motion(const double start_time, const double end_time)const=0;

  virtual RigidObject3DTransformation 
    compute_average_motion_rel_time(const double start_time, const double end_time)const = 0;

  virtual void get_motion_rel_time(RigidObject3DTransformation& ro3dtrans, const double time) const =0;

  //! Has to be called and will be used to synchronise listmode time and motion tracking time
  /*! This should make sure that a 'rel_time' of 0 corresponds to the start of the list mode data
   */
  virtual Succeeded synchronise(CListModeData&) =0;


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

  const RigidObject3DTransformation& 
    get_transformation_to_reference_position() const;

  virtual const RigidObject3DTransformation& 
    get_transformation_to_scanner_coords() const = 0;
  virtual const RigidObject3DTransformation& 
    get_transformation_from_scanner_coords() const = 0;

protected:
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  // next member is protected in case it's needed by synchronise()
  string list_mode_filename;

private:
  
  string attenuation_filename; 
  double transmission_duration;
  std::time_t  reference_start_time_in_secs_since_1970;
  std::time_t reference_end_time_in_secs_since_1970;
  vector<double> reference_translation;
  vector<double> reference_quaternion;
  RigidObject3DTransformation transformation_to_reference_position;

};

END_NAMESPACE_STIR
