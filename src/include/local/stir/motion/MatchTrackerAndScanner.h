//
// $Id$
//
/*
    Copyright (C) 2003- $Date$ , Hammersmith Imanet Ltd
    For GE Internal use only
*/

#ifndef __stir_motion_MatchTrackerAndScanner_H__
#define __stir_motion_MatchTrackerAndScanner_H__
/*!
  \file
  \ingroup motion
  \brief Definition of class stir::MatchTrackerAndScanner.

  \author Kris Thielemans

  
  $Date$
  $Revision$
*/
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/ParsingObject.h"
#include "local/stir/motion/RigidObject3DMotion.h"

START_NAMESPACE_STIR
/*! \ingroup motion
  \brief A class for finding the coordinate transformation between tracker and scanner coordinate systems

  \par Input data

  You need to have performed a scan where the '0' marker of the tracker is filled with
  some radioactive material, and the marker is then moved to various stationary positions inside
  the scanner, while performing a (list-mode) scan, while the tracker is running. 
  Remember: discrete movements. 
  
  Then you need to make a frame-definition file where all non-stationary parts are skipped.
  Then you sort the list mode data into sinograms, and reconstruct images (preferably with large zoom)
  for each of these discrete positions.
  
  \par What does it do?

  This implements Horn's method, as discussed in detail in Roger Fulton's thesis. Briefly:
  - for each time frame, we find the centre of gravity (after applying
    a threshold) in the image as the location of the point source.
  - for each tracker-sample in the time frame, we find where the tracker says that the
    0 marker moved to.
  - All the data for all time frames is then put through 
    RigidObject3DTransformation::find_closest_transformation to find a least-squares fit
    between the 2 sets of coordinates.
  - The class writes various diagnostics to stdout, including the value of the fit.

  Note that if you have movement within a time frame, the diagnostics will tell you, but
  your fit will be wrong.

  \par Example par file
  \verbatim
  MoveImage Parameters:=
  ; see TimeFrameDefinitions
  time frame_definition filename := frame_definition_filename
  
  ; next parameter is optional (and not normally necessary)
  ; it can be used if the frame definitions are relative to another scan as what 
  ; is used to for the rigid object motion (i.e. currently the list mode data used
  ;  for the Polaris synchronisation)
  ; scan_start_time_secs_since_1970_UTC


  ; specify motion, see stir::RigidObject3DMotion
  Rigid Object 3D Motion Type := type

  ; optional field to determine relative threshold to apply to 
  ; the image before taking the centre of gravity
  ; it is relative to the maximum in each image (i.e. .5 would be at half the maximum)
  ; default is .1
  relative threshold := .2
  ; prefix for finding the images
  ; filenames will be constructed by appending _f#g1d0b0 (and the extension .hv)
  image_filename_prefix :=
  END :=
\endverbatim

  \warning Currently the motion object needs to be defined using a
  transformation_from_scanner_coords file. However, the value of the transformation
  is completely ignored by the current class.
*/  
class MatchTrackerAndScanner : public ParsingObject
{
public:
  MatchTrackerAndScanner(const char * const par_filename);

  //! finds the match when all parameters have been set
  /*! will store the transformation as part of this object, but also write it to stdout
   */
  Succeeded run();

  const TimeFrameDefinitions&
    get_time_frame_defs() const;

  double get_frame_start_time(unsigned frame_num) const
  { return frame_defs.get_start_time(frame_num) + scan_start_time; }

  double get_frame_end_time(unsigned frame_num) const
  { return frame_defs.get_end_time(frame_num) + scan_start_time; }

  const string& get_image_filename_prefix() const
  { return _image_filename_prefix; }
  
  const RigidObject3DMotion& get_motion() const
  { return *_ro3d_sptr; }

  const RigidObject3DTransformation& 
    get_transformation_from_scanner_coords() const
  { return _transformation_from_scanner_coords; }

protected:

  // all of these really should be in a AbsTimeFrameDefinitions class or so
  TimeFrameDefinitions frame_defs;
  int scan_start_time_secs_since_1970_UTC;
  double _current_frame_end_time;
  double _current_frame_start_time;
  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  string frame_definition_filename;

  double scan_start_time;

  string _image_filename_prefix;

  float relative_threshold;
private:
  shared_ptr<RigidObject3DMotion> _ro3d_sptr;

  // will be set to new value
  RigidObject3DTransformation _transformation_from_scanner_coords;  
};

END_NAMESPACE_STIR

#endif
