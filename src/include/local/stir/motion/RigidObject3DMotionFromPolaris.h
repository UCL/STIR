//
// $Id: 
//
/*!
  \file
  \ingroup local_buildblock

  \brief Declaration of class RigidObject3DMotionFromPolaris

  \author  Sanida Mustafovic and Kris Thielemans
  $Date: 
  $Revision: 
*/

/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/



#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

class RigidObject3DMotionFromPolaris: public RegisteredParsingObject<RigidObject3DMotionFromPolaris,RigidObject3DMotion> 
{
public:
  //! Name which will be used when parsing a MotionTracking object 
  static const char * const registered_name; 
  // only neeed this to enable LmToProjDataWithMC(const char * const par_filename) function

  RigidObject3DMotionFromPolaris();

  RigidObject3DMotionFromPolaris(const string mt_filename,shared_ptr<Polaris_MT_File> mt_file_ptr);

     // TODO ???
  ~RigidObject3DMotionFromPolaris() {};

  //! Set the reference position 
  virtual RigidObject3DTransformation compute_average_motion(const float start_time, const float end_time) const;

  //! Given the time obtain motion info, i.e. RigidObject3DTransformation
  virtual void get_motion(RigidObject3DTransformation& ro3dtrans, const float time) const;

  //! Synchronise motion tracking file and listmode file
  virtual Succeeded synchronise(const CListModeData& listmode_data);
  
//private: 
  //! Find and store gating values in a vector from lm_file  
  void find_and_store_gate_tag_values_from_lm(vector<float>& lm_time, 
					      vector<unsigned>& lm_random_number,
					      const CListModeData& listmode_data);

  //! Find and store random numbers from mt_file
  void find_and_store_random_numbers_from_mt_file(vector<unsigned>& mt_random_numbers);

  void find_offset(const CListModeData& listmode_data);


  shared_ptr<Polaris_MT_File> mt_file_ptr;
  float Polaris_time_offset;
  string mt_filename;  
  //string lm_filename;

#if 1
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
#endif

};


END_NAMESPACE_STIR
