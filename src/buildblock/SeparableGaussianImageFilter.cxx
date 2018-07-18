
#include "local/stir/SeparableGaussianImageFilter.h"
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR

template <typename elemT>
SeparableGaussianImageFilter<elemT>::
SeparableGaussianImageFilter()
//:filter_coefficients(VectorWithOffset<float>(-1,1))
{
    set_defaults();
}

template <typename elemT>
float
SeparableGaussianImageFilter<elemT>::
get_standard_deviation()
{
  return standard_deviation;
}
  
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

   /*gaussian_filter =
    SeparableGaussianArrayFilter<3,elemT>(standard_deviation,     
                    number_of_coefficients);  */
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
void
SeparableGaussianImageFilter<elemT>::
set_defaults()
{
  //standard_deviation =0 ;
  //number_of_coefficients =0;
    
}

template <typename elemT>
void 
SeparableGaussianImageFilter<elemT>::
initialise_keymap()
{
  this->parser.add_start_key("Separable Gaussian Filter Parameters");
  this->parser.add_key ("standard_deviation", &standard_deviation);
  this->parser.add_key ("number_of_coefficients", &number_of_coefficients);
  this->parser.add_stop_key("END Separable Gaussian Filter Parameters");

}

template <typename elemT>
bool 
SeparableGaussianImageFilter<elemT>::
post_processing()
{
  return false;
  /*const unsigned int size = filter_coefficients_for_parsing.size();
  const int min_index = -(size/2);
  filter_coefficients.grow(min_index, min_index + size - 1);
  for (int i = min_index; i<= filter_coefficients.get_max_index(); ++i)
    filter_coefficients[i] = 
      static_cast<float>(filter_coefficients_for_parsing[i-min_index]);
  return false;*/
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
