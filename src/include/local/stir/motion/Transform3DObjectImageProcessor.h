//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
*/
/*!

  \file
  \ingroup ImageProcessor  
  \ingroup motion
  \brief Declaration of class stir::Transform3DObjectImageProcessor
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/

#ifndef __stir_motion_Transform3DObjectImageProcessor_H__
#define __stir_motion_Transform3DObjectImageProcessor_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/ImageProcessor.h"
#include "stir/shared_ptr.h"
#include "local/stir/motion/ObjectTransformation.h"
// next is currently needed to get Array<pair<>> to compile (definition of assign() in there)
#include "local/stir/motion/transform_3d_object.h"


START_NAMESPACE_STIR

// TODO!! remove define

#define num_dimensions 3

/*!
  \ingroup ImageProcessor  
  \brief A class in the ImageProcessor hierarchy that performs movement by reinterpolation
  \warning This class is currently restricted to 3d. 
 */

template <typename elemT>
class Transform3DObjectImageProcessor : 
  public 
    RegisteredParsingObject<
        Transform3DObjectImageProcessor<elemT>,
        ImageProcessor<3,elemT>,
        ImageProcessor<3,elemT>
    >
{
public:
  static const char * const registered_name; 
  
  //! Default constructor
  Transform3DObjectImageProcessor();
  
  
private:
  //motion
  shared_ptr<ObjectTransformation<3,float> > transformation_sptr;
  bool _do_transpose;
  bool _do_jacobian;  
  bool _cache_transformed_coords;

  Array<3, BasicCoordinate<3,float> > _transformed_coords;
  Array<3, std::pair<BasicCoordinate<3,float>, float> > _transformed_coords_and_jacobian;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing(); 
  
  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& image);
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};

#undef num_dimensions

END_NAMESPACE_STIR

#endif


