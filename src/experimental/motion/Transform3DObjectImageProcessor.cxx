//
//
/*
    Copyright (C) 2005- 2012, Hammersmith Imanet Ltd
*/
/*!
  \file
  \ingroup ImageProcessor  
  \brief Implementation of class stir::Transform3DObjectImageProcessor
    
  \author Kris Thielemans

*/

#include "stir_experimental/motion/Transform3DObjectImageProcessor.h"
#include "stir_experimental/motion/transform_3d_object.h"
#include "stir_experimental/numerics/more_interpolators.h"
#include "stir/is_null_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/stream.h"
#include "stir/CPUTimer.h"
#include "stir/info.h"
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
  base_type::initialise_keymap();

  this->parser.add_start_key("Transformation Parameters");
#if XXX
  this->parser.add_key("transformation",&this->transformation_as_string);
#else
  this->parser.add_parsing_key("transformation type",&this->transformation_sptr);
#endif
  this->parser.add_key("do jacobian", &this->_do_jacobian);
  this->parser.add_key("do transpose", &this->_do_transpose);
  this->parser.add_key("cache_transformed_coords", &this->_cache_transformed_coords);
  this->parser.add_stop_key("END Transformation Parameters");

}

template <typename elemT>
bool 
Transform3DObjectImageProcessor<elemT>::
post_processing()
{
  if (base_type::post_processing() != false)
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
  info(boost::format("'transformation' quaternion  %1%") % this->transformation.get_quaternion());
  info(boost::format("'transformation' translation  %1%") % this->transformation.get_translation());
#else
  if (is_null_ptr(transformation_sptr))
    {
      warning("No transformation set");
      return true;
    }
#endif
  this->_transformed_coords.recycle();
  return false;
}

template <typename elemT>
Transform3DObjectImageProcessor<elemT>::
Transform3DObjectImageProcessor(const shared_ptr<ObjectTransformation<3,elemT> > transf)
{
  set_defaults();
  this->transformation_sptr = transf;
}

template <typename elemT>
Succeeded
Transform3DObjectImageProcessor<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{
  if (this->_cache_transformed_coords)
    {
      CPUTimer timer;
      timer.start();
      if (!this->_do_jacobian )
	{
	  this->_transformed_coords =
	    find_grid_coords_of_transformed_centres(density, 
						    density, 
						    *this->transformation_sptr);
	}
      else
	{
	  this->_transformed_coords_and_jacobian =
	    find_grid_coords_of_transformed_centres_and_jacobian(density, 
								 density, 
								 *this->transformation_sptr);
	}
      timer.stop();
      info(boost::format("CPU time for computing centre coords %1% secs") % timer.value());
    }
  else
    {
      this->_transformed_coords.recycle();
      this->_transformed_coords_and_jacobian.recycle();
    }
  return Succeeded::yes;  
}


template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{ 
  shared_ptr<DiscretisedDensity<3,elemT> > density_copy_sptr(density.clone());
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

  if (this->_cache_transformed_coords)
    {
      if (this->_do_transpose)
	{
	  PushTransposeLinearInterpolator<float> interpolator;
	  //PushNearestNeighbourInterpolator<float> interpolator;
	  interpolator.set_output(out_density);

	  for (int z= in_density.get_min_index(); z<= in_density.get_max_index(); ++z)
	    for (int y= in_density[z].get_min_index(); y<= in_density[z].get_max_index(); ++y)
	      for (int x= in_density[z][y].get_min_index(); x<= in_density[z][y].get_max_index(); ++x)
		{
		  if (this->_do_jacobian )
		    {
		      const float jacobian =
			this->_transformed_coords_and_jacobian[z][y][x].second;
		      if (jacobian<.01)
			error("jacobian too small : %g at z=%d,y=%d,x=%d",
			      jacobian,z,y,x);
		      interpolator.add_to(this->_transformed_coords_and_jacobian[z][y][x].first,
					  in_density[z][y][x]*jacobian);
		    }
		  else
		    interpolator.add_to(this->_transformed_coords[z][y][x], in_density[z][y][x]);
		}
	}
      else
	{
	  PullLinearInterpolator<float> interpolator;
	  //PullNearestNeighbourInterpolator<float> interpolator;
	  interpolator.set_input(in_density);
	  for (int z= out_density.get_min_index(); z<= out_density.get_max_index(); ++z)
	    for (int y= out_density[z].get_min_index(); y<= out_density[z].get_max_index(); ++y)
	      for (int x= out_density[z][y].get_min_index(); x<= out_density[z][y].get_max_index(); ++x)
		{
		  if (this->_do_jacobian )
		    {
		      const float jacobian =
			this->_transformed_coords_and_jacobian[z][y][x].second;
		      if (jacobian<.01)
			error("jacobian too small : %g at z=%d,y=%d,x=%d",
			      jacobian,z,y,x);
		      // TODO thnk about divide or multiply jacobian
		      out_density[z][y][x] =
			interpolator(this->_transformed_coords_and_jacobian[z][y][x].first)*jacobian;
		    }
		  else
		    out_density[z][y][x] =
		      interpolator(this->_transformed_coords[z][y][x]);
		}
	}
    }
  else // !_cache_transformed_coords
    {
      if (this->_do_transpose)
	transform_3d_object_push_interpolation(out_density,
					       in_density,
					       *this->transformation_sptr,
					       PushTransposeLinearInterpolator<float>(),
					       //PushNearestNeighbourInterpolator<float>(),
					       this->_do_jacobian);
      else
	transform_3d_object_pull_interpolation(out_density,
					       in_density,
					       *this->transformation_sptr,
					       PullLinearInterpolator<float>(),
					       //PullNearestNeighbourInterpolator<float>(),
					       this->_do_jacobian );
    }
#endif
}


template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
set_defaults()
{
  base_type::set_defaults();
  this->_do_transpose=false;
  this->_do_jacobian=false;
  this->_cache_transformed_coords=false;
#if XXX
  this->transformation = RigidObject3DTransformation(Quaternion<float>(1,0,0,0), 
					       CartesianCoordinate3D<float>(0,0,0));
#else
  this->transformation_sptr.reset();
#endif
}


template <typename elemT>
bool
Transform3DObjectImageProcessor<elemT>::
get_do_transpose() const
{
  return this->_do_transpose;
}

template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
set_do_transpose(const bool value)
{
  this->_do_transpose = value;
}

template <typename elemT>
bool
Transform3DObjectImageProcessor<elemT>::
get_do_jacobian() const
{
  return this->_do_jacobian;
}

template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
set_do_jacobian(const bool value)
{
  this->_do_jacobian = value;
}

template <typename elemT>
bool
Transform3DObjectImageProcessor<elemT>::
get_do_cache() const
{
  return this->_cache_transformed_coords;
}

template <typename elemT>
void
Transform3DObjectImageProcessor<elemT>::
set_do_cache(const bool value)
{
  this->_cache_transformed_coords = value;
}

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class Transform3DObjectImageProcessor<float>;

END_NAMESPACE_STIR
