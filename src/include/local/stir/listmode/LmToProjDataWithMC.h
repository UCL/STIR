//
// $Id$
//
/*!

  \file
  \ingroup listmode
  \brief Class for rebinning listmode files with motion correction
    
  \author Sanida Mustafovic
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_LmToProjDataWithMC_H__
#define __stir_listmode_LmToProjDataWithMC_H__


#include "stir/listmode/LmToProjData.h"
#include "stir/CartesianCoordinate3D.h"
#include "local/stir/AbsTimeInterval.h"
#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"

START_NAMESPACE_STIR

class LmToProjDataWithMC : public LmToProjData
{
public:
     
  LmToProjDataWithMC(const char * const par_filename);

  virtual void get_bin_from_event(Bin& bin, const CListEvent&) const;
  virtual void process_new_time_event(const CListTime& time_event);

private:

  shared_ptr<RigidObject3DMotion> ro3d_ptr;
   
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  RigidObject3DTransformation move_to_scanner;
  RigidObject3DTransformation move_from_scanner;
  shared_ptr<AbsTimeInterval> _reference_abs_time_sptr;
  
  RigidObject3DTransformation _transformation_to_reference_position;
  RigidObject3DTransformation ro3dtrans; // actual Polaris  motion for current_time
  



  
};

END_NAMESPACE_STIR


#endif
