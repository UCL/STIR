//
// $Id$
//
/*!
  \file
  \ingroup motion

  \brief Declaration of class RigidObject3DMotion

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$
*/

/*
    Copyright (C) 2003- $Date$ , Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/listmode/CListModeData.h"
#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"


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

  virtual RigidObject3DTransformation 
    compute_average_motion(const double start_time, const double end_time)const=0;

  RigidObject3DTransformation 
    compute_average_motion_rel_time(const double start_time, const double end_time)const;

  virtual void get_motion_rel_time(RigidObject3DTransformation& ro3dtrans, const double time) const =0;

  //! Has to set \a time_offset and will be used to synchronise listmode time and motion tracking time
  virtual Succeeded synchronise(CListModeData&) =0;
 protected:
  //!  Option to set  time offset manually in case synchronisation cannot be performed
  void
    set_time_offset(const double time_offset);
  double 
    get_time_offset() const;
  //! Temporary function to allow others to see if they should call synchronise or not
  bool is_synchronised() const;
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

  double time_offset;
  // next member is protected in case it's needed by synchronise()
  string list_mode_filename;

private:

  static void 
    find_ref_start_end_from_att_file (double& att_start_time, double& att_end_time, 
				      const double transmission_duration, 
				      const string& attnuation_filename);
 
  
  string attenuation_filename; 
  double transmission_duration;
  double reference_start_time;
  double reference_end_time;
  vector<double> reference_translation;
  vector<double> reference_quaternion;
  RigidObject3DTransformation transformation_to_reference_position;

};

END_NAMESPACE_STIR
