//
// $Id: 
//
/*!
  \file
  \ingroup local_buildblock

  \brief Declaration of class RigidObject3DMotion

  \author  Sanida Mustafovic and Kris Thielemans
  $Date: 
  $Revision: 
*/

/*
    Copyright (C) 2000- $Date: , IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/listmode/CListModeData.h"
#include "stir/RegisteredObject.h"


START_NAMESPACE_STIR


class RigidObject3DMotion: public RegisteredObject<RigidObject3DMotion>
{

public:
  virtual RigidObject3DTransformation compute_average_motion(const float start_time, const float end_time)const=0;

  virtual void get_motion(RigidObject3DTransformation& ro3dtrans, const float time) const =0;

  virtual Succeeded synchronise(const CListModeData&) =0;
  
protected:

};

END_NAMESPACE_STIR
