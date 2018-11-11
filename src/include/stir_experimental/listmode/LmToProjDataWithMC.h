//
//
/*
    Copyright (C) 2003- 2005, Hammersmith Imanet
    For GE internal use only
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::LmToProjDataWithMC

    
  \author Sanida Mustafovic
  \author Kris Thielemans
      
*/

#ifndef __stir_listmode_LmToProjDataWithMC_H__
#define __stir_listmode_LmToProjDataWithMC_H__


#include "stir/listmode/LmToProjData.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir_experimental/AbsTimeInterval.h"
#include "stir_experimental/motion/RigidObject3DMotionFromPolaris.h"

START_NAMESPACE_STIR

/*! \ingroup listmode
  \brief Class for binning list mode files with motion correction

  Implements LOR repositioning during bining of list mode data into sinograms.

*/
class LmToProjDataWithMC : public LmToProjData
{
public:
     
  LmToProjDataWithMC(const char * const par_filename);

  virtual void get_bin_from_event(Bin& bin, const CListEvent&) const;
  virtual void process_new_time_event(const CListTime& time_event);

protected: 
  //! motion information
  shared_ptr<RigidObject3DMotion> ro3d_ptr;
  //! switch between constant reference position, or one for each frame
  bool reference_position_is_average_position_in_frame;
  //! constant reference position (if used)
  shared_ptr<AbsTimeInterval> _reference_abs_time_sptr;  

  virtual void start_new_time_frame(const unsigned int new_frame_num);

   
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

private:

  RigidObject3DTransformation _transformation_to_reference_position;

  RigidObject3DTransformation ro3dtrans; // actual  motion for current_time
    
};

END_NAMESPACE_STIR


#endif
