//
// $Id$
//
/*!
  \file
  \ingroup motion

  \brief Declaration of class RigidObject3DMotionFromPolaris

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$ 
*/

/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR
/*! \ingroup motion
  A class for handling motion information from the Polaris tracker
*/
class RigidObject3DMotionFromPolaris: 
  public 
  RegisteredParsingObject< RigidObject3DMotionFromPolaris,
			   RigidObject3DMotion,
			   RigidObject3DMotion>
                                                             
{
public:
  //! Name which will be used when parsing a MotionTracking object 
  static const char * const registered_name; 

  // only need this to enable LmToProjDataWithMC(const char * const par_filename) function
  RigidObject3DMotionFromPolaris();

  //! Find average motion from the Polaris file 
  virtual RigidObject3DTransformation 
    compute_average_motion(const double start_time, const double end_time) const;

  //! Given the time obtain motion info, i.e. RigidObject3DTransformation
  virtual void get_motion_rel_time(RigidObject3DTransformation& ro3dtrans, const double time) const;

  //! Synchronise motion tracking file and listmode file
  virtual Succeeded synchronise(CListModeData& listmode_data);

  virtual const RigidObject3DTransformation& 
    get_transformation_to_scanner_coords() const;
  virtual const RigidObject3DTransformation& 
    get_transformation_from_scanner_coords() const;
  
private: 

  void do_synchronisation(CListModeData& listmode_data);


  shared_ptr<Polaris_MT_File> mt_file_ptr;
  string mt_filename;  
 
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

 private:
  // TODO this should probably be moved to RigidObject3DMotion
  string transformation_from_scanner_coordinates_filename;
  RigidObject3DTransformation move_to_scanner_coords;
  RigidObject3DTransformation move_from_scanner_coords;

};


END_NAMESPACE_STIR
