//
//
/*
    Copyright (C) 2005- 2012, Hammersmith Imanet Ltd
*/
/*!

  \file
  \ingroup ImageProcessor  
  \ingroup motion
  \brief Declaration of class stir::Transform3DObjectImageProcessor
  \author Kris Thielemans
      
*/

#ifndef __stir_motion_Transform3DObjectImageProcessor_H__
#define __stir_motion_Transform3DObjectImageProcessor_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/shared_ptr.h"
#include "stir_experimental/motion/ObjectTransformation.h"
// next is currently needed to get Array<pair<>> to compile (definition of assign() in there)
#include "stir_experimental/motion/transform_3d_object.h"


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
        DataProcessor<DiscretisedDensity<3,elemT> >,
        DataProcessor<DiscretisedDensity<3,elemT> >
    >
{
  typedef DataProcessor<DiscretisedDensity<3,elemT> > base_type;
public:
  static const char * const registered_name; 
  
  //! Default constructor
  //Transform3DObjectImageProcessor();
  //! Constructor that set the transformation
  explicit
    Transform3DObjectImageProcessor(const shared_ptr<ObjectTransformation<3,elemT> >  = shared_ptr<ObjectTransformation<3,elemT> >());

  bool get_do_transpose() const;
  void set_do_transpose(const bool);
  bool get_do_jacobian() const;
  void set_do_jacobian(const bool);
  bool get_do_cache() const;
  void set_do_cache(const bool);

private:
  //motion
  shared_ptr<ObjectTransformation<3,elemT> > transformation_sptr;
  bool _do_transpose;
  bool _do_jacobian;  
  bool _cache_transformed_coords;

  Array<3, BasicCoordinate<3,elemT> > _transformed_coords;
  Array<3, std::pair<BasicCoordinate<3,elemT>, elemT> > _transformed_coords_and_jacobian;

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


