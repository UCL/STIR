//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class SeparableCartesianMetzImageFilter

  \author Sanida Mustafovic
  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/
#include "tomo/SeparableCartesianMetzImageFilter.h"
#include "VoxelsOnCartesianGrid.h"


START_NAMESPACE_TOMO

  
template <typename elemT>
Succeeded
SeparableCartesianMetzImageFilter<elemT>::virtual_build_filter(const DiscretisedDensity<3,elemT>& density)

{
/*  if (consistency_check(density) == Succeeded::no)
    return Succeeded::no;
  */
  const VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);

  metz_filter = SeparableMetzArrayFilter<3,elemT>(get_metz_fwhm(),get_metz_powers(),image.get_voxel_size());
  
  return Succeeded::yes;
  
}


template <typename elemT>
void
SeparableCartesianMetzImageFilter<elemT>::filter_it(DiscretisedDensity<3,elemT>& density) const

{     
  //assert(consistency_check(density) == Succeeded::yes);
  metz_filter(density);
  
}


template <typename elemT>
void
SeparableCartesianMetzImageFilter<elemT>::filter_it(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const
{
  //assert(consistency_check(in_density) == Succeeded::yes);
  metz_filter(out_density,in_density);
}

#if 0

template <typename elemT>
Succeeded
SeparableCartesianMetzImageFilter<elemT>:: consistency_check( const DiscretisedDensity<3, elemT>& image) const
{
  
  //TODO?
  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);
  
  CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
  
  // checks if metz_powers >= 0, also checks if FWHM of the filter
  // is smaller than a sampling interval(to prevent bandwidth of the filter
  // exceed NF of the image)
  if ( metz_powers[0]>=0 && metz_powers[1]>=0 &&metz_powers[1]>=0&& metz_filter.fwhms[0] <=voxel_size.x() &&  metz_filter.fwhms[1] <=voxel_size.y()&&  metz_filter.fwhms[1] <=voxel_size.z())
    
    warning("Filter's fwhm is smaller than a sampling distance in the image\n");
  return Succeeded::yes;
  //else
  //return Succeeded::no;
}
#endif

template <typename elemT>
SeparableCartesianMetzImageFilter<elemT>::SeparableCartesianMetzImageFilter()
: fwhm(VectorWithOffset<elemT>(1,3)),
  metz_power(VectorWithOffset<elemT>(1,3))
{
  set_defaults();
}

template <typename elemT>
VectorWithOffset<float>
SeparableCartesianMetzImageFilter<elemT>:: get_metz_fwhm()
{  return fwhm;}

template <typename elemT>
VectorWithOffset<float> 
SeparableCartesianMetzImageFilter<elemT>::get_metz_powers()
{  return metz_power;}

template <typename elemT>
void
SeparableCartesianMetzImageFilter<elemT>::set_defaults()
{
/*
   fwhm_xy = 1.;
   fwhm_z = 1.;
   metz_power_xy = 1;
   metz_power_z = 1;
*/
  fwhm.fill(0);
  metz_power.fill(0);  
}

template <typename elemT>
void 
SeparableCartesianMetzImageFilter<elemT>::initialise_keymap()
{
  parser.add_start_key("Separable Cartesian Metz Filter Parameters");

  parser.add_key("x-dir filter FWHM (in mm)", &fwhm[3]);
  parser.add_key("y-dir filter FWHM (in mm)", &fwhm[2]);
  parser.add_key("z-dir filter FWHM (in mm)", &fwhm[1]);
  parser.add_key("x-dir filter Metz power", &metz_power[3]);
  parser.add_key("y-dir filter Metz power", &metz_power[2]);
  parser.add_key("z-dir filter Metz power", &metz_power[1]);   
  parser.add_stop_key("END Separable Cartesian Metz Filter Parameters");
}



const char * const 
SeparableCartesianMetzImageFilter<float>::registered_name =
  "Separable Cartesian Metz";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template SeparableCartesianMetzImageFilter<float>;
END_NAMESPACE_TOMO



