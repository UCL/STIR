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
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/listmode/CListModeData.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/stream.h"
#include "stir/round.h"
#include "stir/is_null_ptr.h"
#include "local/stir/AbsTimeInterval.h"

#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR



void 
RigidObject3DMotion::set_defaults()
{ 
  list_mode_filename="";
  //time_offset=time_not_yet_determined;
}

void 
RigidObject3DMotion::initialise_keymap()
{ 
#if 0
  parser.add_key("reference_quaternion", &reference_quaternion);
  parser.add_key("reference_translation", &reference_translation);
#endif
  parser.add_key("list_mode_filename",&list_mode_filename);
}

bool
RigidObject3DMotion::
post_processing()
{
  if (list_mode_filename.size()!=0)
    {
      shared_ptr<CListModeData> lm_data_ptr =
	CListModeData::read_from_file(list_mode_filename);

      synchronise(*lm_data_ptr);
    }

  if (!is_synchronised())
    {
      warning("RigidObject3DMotion object not synchronised.");
      return true;
    }
#if 0
  else 
    {
      if (reference_translation.size()!=3 || reference_quaternion.size() !=4)
	{
	  warning ("Invalid reference quaternion or translation. Give either attenuation, reference_start/end_time or quaternion/translation\n");
	  return true;
	}
      const CartesianCoordinate3D<float>ref_trans(static_cast<float>(reference_translation[0]),
						  static_cast<float>(reference_translation[1]),
						  static_cast<float>(reference_translation[2]));
      const Quaternion<float>ref_quat(static_cast<float>(reference_quaternion[0]),
				      static_cast<float>(reference_quaternion[1]),
				      static_cast<float>(reference_quaternion[2]),
				      static_cast<float>(reference_quaternion[3]));
      RigidObject3DTransformation av_motion(ref_quat, ref_trans);
      cerr << "Reference quaternion from par file:  " << av_motion.get_quaternion()<<endl;
      cerr << "Reference translation from par file:  " << av_motion.get_translation()<<endl;
      transformation_to_reference_position =av_motion.inverse();
    }
#endif

  return false;
}

#if 0
RigidObject3DTransformation 
RigidObject3DMotion::
compute_average_motion_rel_time(const double start_time, const double end_time) const
{
  return compute_average_motion(start_time + time_offset, end_time + time_offset);
}
#endif

RigidObject3DTransformation
RigidObject3DMotion::
compute_average_motion(const AbsTimeInterval& interval) const
{
  return compute_average_motion_rel_time(secs_since_1970_to_rel_time(interval.get_start_time_in_secs_since_1970()),
					 secs_since_1970_to_rel_time(interval.get_end_time_in_secs_since_1970()));
}

#if 0
void
RigidObject3DMotion::
set_time_offset(const double offset)
{
  time_offset=offset;
}

double
RigidObject3DMotion::
get_time_offset() const
{
  return time_offset;
}

bool 
RigidObject3DMotion::
is_synchronised() const
{
  return time_offset!=time_not_yet_determined;
}  
#endif

END_NAMESPACE_STIR
