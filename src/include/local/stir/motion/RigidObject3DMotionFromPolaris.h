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

  virtual RigidObject3DTransformation 
    compute_average_motion_rel_time(const double start_time, const double end_time) const;

  //! Given the time obtain motion info, i.e. RigidObject3DTransformation
  virtual void get_motion_rel_time(RigidObject3DTransformation& ro3dtrans, const double time) const;

  //! Synchronise motion tracking file and listmode file
  virtual Succeeded synchronise(CListModeData& listmode_data);
  virtual double secs_since_1970_to_rel_time(std::time_t) const;

  virtual const RigidObject3DTransformation& 
    get_transformation_to_scanner_coords() const;
  virtual const RigidObject3DTransformation& 
    get_transformation_from_scanner_coords() const;

  Succeeded set_mt_file(const string& mt_filename);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! Gets boundaries to determine when the time offset is out of bounds
  //*! Currently, the time offset is compared to the start of the listmode scan.*/
  double get_max_time_offset_deviation() const
    { return max_time_offset_deviation; }
  //! Sets boundaries to determine when the time offset is out of bounds
  void set_max_time_offset_deviation(const double v)
    { max_time_offset_deviation = v; }
  //! Gets boundaries to determine when the time drift is too large
  /*! deviation is measured as fabs(time_drift-1) */
  double get_max_time_drift_deviation() const
    { return max_time_drift_deviation; }
  //! Sets boundaries to determine when the time drift is too large
  void set_max_time_drift_deviation(const double v)
    { max_time_drift_deviation = v; }

private: 

  void do_synchronisation(CListModeData& listmode_data);
  virtual bool is_synchronised() const;

  double rel_time_to_polaris_time(const double time) const;

 RigidObject3DTransformation 
  compute_average_motion_polaris_time(const double start_time, const double end_time)const;


  shared_ptr<Polaris_MT_File> mt_file_ptr;
  string mt_filename;  
 

 private:
  // TODO this should probably be moved to RigidObject3DMotion
  string transformation_from_scanner_coordinates_filename;
  RigidObject3DTransformation move_to_scanner_coords;
  RigidObject3DTransformation move_from_scanner_coords;

  double time_offset;
  double time_drift;

  //! A variable used to determine when the time offset is out of bounds
  //*! Currently, the time offset is compared to the start of the listmode scan.*/
  double max_time_offset_deviation;
  //! A variable used to determine when the time drift is too large
  /*! deviation is measured as fabs(time_drift-1) */
  double max_time_drift_deviation;

  std::time_t listmode_data_start_time_in_secs;
};


END_NAMESPACE_STIR
