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

#include <ctime>

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; using ::tm; using ::localtime; }
#endif

#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

static const double time_not_yet_determined=-1234567.8;

void 
RigidObject3DMotion::
find_ref_start_end_from_att_file (double& att_start_time, double& att_end_time, 
			    double transmission_duration,
			    const string& attenuation_filename)
{
#ifdef HAVE_LLN_MATRIX
  MatrixFile* AttnFile = matrix_open(attenuation_filename.c_str(), MAT_READ_ONLY, AttenCor );
  if (AttnFile==NULL)
    error("Error opening attenuation file %s\n", attenuation_filename.c_str());

  /* Acquisition date and time - main head */
  std::time_t sec_time = AttnFile->mhptr->scan_start_time;

  struct std::tm* AttnTime = std::localtime( &sec_time  ) ;
  matrix_close( AttnFile ) ;
  att_start_time = ( AttnTime->tm_hour * 3600.0 ) + ( AttnTime->tm_min * 60.0 ) + AttnTime->tm_sec ;
  att_end_time = att_start_time + transmission_duration;
#else
    error("Error opening attenuation file %s: compiled without ECAT7 support\n", attenuation_filename.c_str());
#endif
}

void 
RigidObject3DMotion::set_defaults()
{ 
  transmission_duration = 300;
  attenuation_filename ="";
  list_mode_filename="";
  reference_start_time=time_not_yet_determined;
  reference_end_time=time_not_yet_determined*10;
  time_offset=time_not_yet_determined;
}

void 
RigidObject3DMotion::initialise_keymap()
{ 
  parser.add_key("attenuation_filename", &attenuation_filename);
  parser.add_key("transmission_duration", &transmission_duration);
  parser.add_key("reference_quaternion", &reference_quaternion);
  parser.add_key("reference_translation", &reference_translation);
  parser.add_key("reference_start_time", &reference_start_time);
  parser.add_key("reference_end_time", &reference_end_time);
  parser.add_key("time_offset", &time_offset);  
  parser.add_key("list_mode_filename",&list_mode_filename);
}

bool
RigidObject3DMotion::
post_processing()
{
  /* complicated way of setting reference motion:
     First try attenuation file. If that fails, try values from 
     reference_start_time/reference_end_time keywords. As a last resort,
     try values from reference_quaternion/reference_translation keywords. */
  if (attenuation_filename !="")
    {
      find_ref_start_end_from_att_file (reference_start_time, reference_end_time,
					transmission_duration,
					attenuation_filename);
      cerr << "reference times from attenuation file: "
	   <<  reference_start_time << " till " << reference_end_time << '\n';
    }
  if (reference_start_time != time_not_yet_determined && 
      reference_start_time < reference_end_time)
    {
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

  if (list_mode_filename.size()!=0)
    {
      shared_ptr<CListModeData> lm_data_ptr =
	CListModeData::read_from_file(list_mode_filename);

      synchronise(*lm_data_ptr);
    }

  if (!is_synchronised())
    {
      warning("RigidObject3DMotion object not synchronised.");
      return false;
    }
  return false;
}

RigidObject3DTransformation 
RigidObject3DMotion::
compute_average_motion_rel_time(const double start_time, const double end_time) const
{
  return compute_average_motion(start_time + time_offset, end_time + time_offset);
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
is_synchronised() const
{
  return time_offset!=time_not_yet_determined;
}  

END_NAMESPACE_STIR
