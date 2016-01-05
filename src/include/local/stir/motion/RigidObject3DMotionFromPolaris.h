//
//
/*
    Copyright (C) 2003- 2010, Hammersmith Imanet Ltd
    For internal GE use only
*/
#ifndef __stir_motion_RigidObject3DMotionFromPolaris__H__
#define __stir_motion_RigidObject3DMotionFromPolaris__H__
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::RigidObject3DMotionFromPolaris

  \author  Sanida Mustafovic and Kris Thielemans
*/

#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/listmode/CListModeData.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR
/*! \ingroup motion
  \brief A class for handling motion information from the Polaris tracker

  Implements synchronisation by comparing gate data in the list mode file
  with the numbers stored in the Polaris .mt file. Computes both time
  offset and clock drift. Synchronisation is stored in a <tt>.sync</tt> file.
  If present, this file is read and actual synchronisation is skipped.
  (This is useful, as the synchronisation is slow as it walks through the
  whole list mode file).

  \par Parsing info
  The class can parse a file for the coordinate transformation that goes from
  scanner to tracker coordinates. This file has to have the following
  format.
\verbatim
  Move from scanner to tracker coordinates:=
   conventions := q0qzqyqx and left-handed
   transformation:= \
    {{-0.00525584, -0.0039961, 0.999977,0.00166456},\
     {-1981.93, 20.1226, 3.96638}}
 end:=
\endverbatim
  Obviously the numbers will depend on your systems.

  \todo move synchronisation out of this class
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

  //! Convert from Polaris transformation to STIR conventions
  /* see more info in .cxx file */
  static 
    RigidObject3DTransformation
    make_transformation_from_polaris_data(Polaris_MT_File::Record const& record);


  // only need this to enable LmToProjDataWithMC(const char * const par_filename) function
  RigidObject3DMotionFromPolaris();

  virtual RigidObject3DTransformation 
    compute_average_motion_in_tracker_coords_rel_time(const double start_time, const double end_time) const;

  virtual 
    RigidObject3DTransformation
    get_motion_in_tracker_coords_rel_time(const double time) const;

  virtual std::vector<double>
    get_rel_time_of_samples(const double start_time, const double end_time) const;

  //! set mask to be able to ignore one or more channels in the listmode gating data
  void set_mask_for_tags(const unsigned int mask_for_tags);

  //! Synchronise motion tracking file and listmode file
  virtual Succeeded synchronise();
  virtual double secs_since_1970_to_rel_time(std::time_t) const;

  virtual const RigidObject3DTransformation& 
    get_transformation_to_scanner_coords() const;
  virtual const RigidObject3DTransformation& 
    get_transformation_from_scanner_coords() const;
  virtual void  
    set_transformation_from_scanner_coords(const RigidObject3DTransformation&);
  Succeeded set_mt_file(const std::string& mt_filename);
  Succeeded set_list_mode_data_file(const std::string& lm_filename);

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
  double polaris_time_to_rel_time(const double time) const;

 RigidObject3DTransformation 
  compute_average_motion_polaris_time(const double start_time, const double end_time)const;


  shared_ptr<Polaris_MT_File> mt_file_ptr;
  std::string mt_filename;  
  std::string list_mode_filename;

 

 private:
  //! allow masking out certain bits of the tags in case a cable is not connected
  unsigned int _mask_for_tags;

  // TODO this should probably be moved to RigidObject3DMotion
  std::string transformation_from_scanner_coordinates_filename;
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
#endif
