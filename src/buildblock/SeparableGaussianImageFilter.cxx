
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR

//TODO remove define

#define num_dimensions 3


template <typename elemT>
Succeeded
SeparableGaussianImageFilter<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{


  if(dynamic_cast<const VoxelsOnCartesianGrid<float>*>(&density)== 0)
    {
      warning("SeparableGaussianImageFilter can only handle images of type VoxelsOnCartesianGrid");
      return Succeeded::no;
    }


  const BasicCoordinate< num_dimensions,float> rescale = dynamic_cast<const VoxelsOnCartesianGrid<float>*>(&density)->get_grid_spacing();


  gaussian_filter = SeparableGaussianArrayFilter<num_dimensions,elemT>(fwhms/rescale,max_kernel_sizes,normalise);

  return Succeeded::yes;

}


template <typename elemT>
void
SeparableGaussianImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{


 gaussian_filter(density);

}


template <typename elemT>
void
SeparableGaussianImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density,
      const DiscretisedDensity<3,elemT>& in_density) const
{

  gaussian_filter(out_density,in_density);
}


template <typename elemT>
SeparableGaussianImageFilter<elemT>::
SeparableGaussianImageFilter()
: fwhms(0),max_kernel_sizes(0)
{
  set_defaults();
}

template <typename elemT>
BasicCoordinate< num_dimensions,float>
SeparableGaussianImageFilter<elemT>::
get_fwhms()
{
  return fwhms;
}

template <typename elemT>
bool
SeparableGaussianImageFilter<elemT>::
get_normalised_filter()
{
    return normalise;
}


template <typename elemT>
BasicCoordinate< num_dimensions,int>
SeparableGaussianImageFilter<elemT>::
get_max_kernel_sizes()
{
  return max_kernel_sizes;
}


template <typename elemT>
void
SeparableGaussianImageFilter<elemT>::
set_defaults()
{
   base_type::set_defaults();
   fwhms.fill(0);
   max_kernel_sizes.fill(-1);
   normalise = false;

}

template <typename elemT>
void
SeparableGaussianImageFilter<elemT>::
initialise_keymap()
{
    base_type::initialise_keymap();
    this->parser.add_start_key("Separable Gaussian Filter Parameters");
    this->parser.add_key("x-dir filter FWHM (in mm)", &fwhms[3]);
    this->parser.add_key("y-dir filter FWHM (in mm)", &fwhms[2]);
    this->parser.add_key("z-dir filter FWHM (in mm)", &fwhms[1]);
    this->parser.add_key("x-dir maximum kernel size", &max_kernel_sizes[3]);
    this->parser.add_key("y-dir maximum kernel size", &max_kernel_sizes[2]);
    this->parser.add_key("z-dir maximum kernel size", &max_kernel_sizes[1]);
    this->parser.add_key("Normalise filter to 1", &normalise);
    this->parser.add_stop_key("END Separable Gaussian Filter Parameters");
}

  


template<>
const char * const 
SeparableGaussianImageFilter<float>::registered_name =
  "Separable Gaussian";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static SeparableGaussianImageFilter<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template class SeparableGaussianImageFilter<float>;

END_NAMESPACE_STIR
