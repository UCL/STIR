//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
*/
/*!
  \file
  \ingroup ImageProcessor  
  \brief Implementation of class stir::Transform3DObjectImageProcessor
    
  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "local/stir/motion/Transform3DObjectImageProcessor.h"
#include "local/stir/motion/transform_3d_object.h"
#include "local/stir/numerics/more_interpolators.h"
#include "stir/is_null_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/stream.h"

START_NAMESPACE_STIR

template<>
const char * const 
Transform3DObjectImageProcessor<float>::registered_name =
#if XXX
"rigid transformation";
#else
"transformation";
#endif


template <typename elemT>
void 
Transform3DObjectImageProcessor<elemT>::
initialise_keymap()
{
  ImageProcessor<3, elemT>::initialise_keymap();

  this->parser.add_start_key("Transformation Parameters");
#if XXX
  this->parser.add_key("transformation",&this->transformation_as_string);
#else
  this->parser.add_parsing_key("transformation type",&this->transformation_sptr);
#endif
  this->parser.add_key("do transpose", &this->_do_transpose);
  this->parser.add_stop_key("END Transformation Parameters");

}

template <typename elemT>
bool 
Transform3DObjectImageProcessor<elemT>::
post_processing()
{
  if (ImageProcessor<3, elemT>::post_processing() != false)
    return true;

#if XXX
  if (this->transformation_as_string.size()>0)
    {
      std::stringstream transformation_as_stream(transformation_as_string);
      transformation_as_stream >> this->transformation;
      if (!transformation_as_stream.good())
	{
	  warning("value for 'transformation' keyword is invalid."
		  "\nIt should be something like '{{q0,qz,qy,qx},{tz,ty,tx}}'");
	  return true;
	}
      if (std::fabs(norm(this->transformation.get_quaternion())-1)>.01)
	{
	  warning("Quaternion should have norm 1 but is %g",
		  norm(this->transformation.get_quaternion()));
	  return true;
	}
    } 
  cerr << "'transformation' quaternion  " << this->transformation.get_quaternion()<<endl;
  cerr << "'transformation' translation  " << this->transformation.get_translation()<<endl;
#else
  if (is_null_ptr(transformation_sptr))
    {
      warning("No transformation set");
      return true;
    }
#endif
  return false;
}

template <typename elemT>
Transform3DObjectImageProcessor<elemT>::
Transform3DObjectImageProcessor()
{
  set_defaults();
}

template <typename elemT>
Succeeded
Transform3DObjectImageProcessor<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{
  return Succeeded::yes;  
}


template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{ 
  shared_ptr<DiscretisedDensity<3,elemT> > density_copy_sptr =
    density.clone();
  density.fill(0);
  this->virtual_apply(density, *density_copy_sptr);  
}


template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density, 
	  const DiscretisedDensity<3,elemT>& in_density) const
{

#if XXX
  if (this->_do_transpose)
    transpose_of_transform_3d_object(out_density, 
				     in_density, 
				     this->transformation);
  else
    transform_3d_object(out_density, 
			in_density, 
			this->transformation);
#else
  if (this->_do_transpose)
    transform_3d_object_push_interpolation(out_density,
					   in_density,
					   *this->transformation_sptr,
					   PushTransposeLinearInterpolator<float>() );
  else
    transform_3d_object_pull_interpolation(out_density,
					   in_density,
					   *this->transformation_sptr,
					   PullLinearInterpolator<float>() );
#endif
}


template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
set_defaults()
{
  ImageProcessor<3, elemT>::set_defaults();
  this->_do_transpose=false;
#if XXX
  this->transformation = RigidObject3DTransformation(Quaternion<float>(1,0,0,0), 
					       CartesianCoordinate3D<float>(0,0,0));
#else
  this->transformation_sptr = 0;
#endif
}


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class Transform3DObjectImageProcessor<float>;

END_NAMESPACE_STIR
