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
#include "stir/IO/stir_ecat7.h"
#include "stir/stream.h"

#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

static const double time_offset_not_yet_determined=-1234567.8;

void 
RigidObject3DMotion::
find_ref_start_end_from_att_file (double& att_start_time, double& att_end_time, 
			    double transmission_duration,
			    const string& attenuation_filename)
{
  MatrixFile* AttnFile = matrix_open(attenuation_filename.c_str(), MAT_READ_ONLY, AttenCor );
  if (AttnFile==NULL)
    error("Error opening attenuation file %s\n", attenuation_filename.c_str());

  /* Acquisition date and time - main head */
  time_t sec_time = AttnFile->mhptr->scan_start_time;

  struct tm* AttnTime = localtime( &sec_time  ) ;
  matrix_close( AttnFile ) ;
  att_start_time = ( AttnTime->tm_hour * 3600.0 ) + ( AttnTime->tm_min * 60.0 ) + AttnTime->tm_sec ;
  att_end_time = att_start_time + transmission_duration;
}

void 
RigidObject3DMotion::set_defaults()
{ 
  transmission_duration = 300;
  attenuation_filename ="";
  reference_start_time=0;
  reference_end_time=0;
  time_offset=time_offset_not_yet_determined;
}

void 
RigidObject3DMotion::initialise_keymap()
{ 
  parser.add_key("attenuation_filename", &attenuation_filename);
  parser.add_key("transmission_duration", &transmission_duration);
  parser.add_key("reference_quaternion", &reference_quaternion);
  parser.add_key("reference_translation", &reference_translation);
  parser.add_key("time_offset", &time_offset);  
}

bool
RigidObject3DMotion::
post_processing()
{
  if (attenuation_filename !="")
    {
      find_ref_start_end_from_att_file (reference_start_time, reference_end_time,
					transmission_duration,
					attenuation_filename);
      cerr << "reference times from attenuation file: "
	   <<  reference_start_time << " till " << reference_end_time << '\n';
      RigidObject3DTransformation av_motion = 
	compute_average_motion(reference_start_time,reference_end_time);
      cerr << "Reference quaternion:  " << av_motion.get_quaternion()<<endl;
      cerr << "Reference translation:  " << av_motion.get_translation()<<endl;
      transformation_to_reference_position =av_motion.inverse();
    
    }
  else 
    {
      if (reference_translation.size()!=3 || reference_quaternion.size() !=4)
	{
	  warning ("Invalid reference quaternion or translation\n");
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
      cerr << "Reference quaternion:  " << av_motion.get_quaternion()<<endl;
      cerr << "Reference translation:  " << av_motion.get_translation()<<endl;
      transformation_to_reference_position =av_motion.inverse();
    }
  return false;
}

const RigidObject3DTransformation &
RigidObject3DMotion::get_transformation_to_reference_position() const
{ 
 return transformation_to_reference_position;
}


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
is_time_offset_set() const
{
  return time_offset!=time_offset_not_yet_determined;
}  

END_NAMESPACE_STIR
