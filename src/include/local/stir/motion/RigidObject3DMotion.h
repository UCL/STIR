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
    Copyright (C) 2003- $Date$ , Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/


#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/listmode/CListModeData.h"
#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"


START_NAMESPACE_STIR


class RigidObject3DMotion: public RegisteredObject<RigidObject3DMotion>,
                           public ParsingObject
{

public:
  virtual ~RigidObject3DMotion() {}

  virtual RigidObject3DTransformation 
    compute_average_motion(const double start_time, const double end_time)const=0;

  RigidObject3DTransformation 
    compute_average_motion_rel_time(const double start_time, const double end_time)const;

  virtual void get_motion(RigidObject3DTransformation& ro3dtrans, const double time) const =0;

  virtual Succeeded synchronise(CListModeData&) =0;
  //!  Option to set  time offset manually in case synchronisation cannot be performed
  void
    set_time_offset(const double time_offset);
  double 
    get_time_offset() const;
  //! Temporary function to allow others to see if they should call synchronise or not
  bool is_time_offset_set() const;

  const RigidObject3DTransformation& 
    get_transformation_to_reference_position() const;

protected:
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  double time_offset;

private:

  static void 
    find_ref_start_end_from_att_file (double& att_start_time, double& att_end_time, 
				      const double transmission_duration, 
				      const string& attnuation_filename);
 
  
  string attenuation_filename; 
  string list_mode_filename;
  double transmission_duration;
  double reference_start_time;
  double reference_end_time;
  vector<double> reference_translation;
  vector<double> reference_quaternion;
  RigidObject3DTransformation transformation_to_reference_position;

};

END_NAMESPACE_STIR
