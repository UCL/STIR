
#include "stir_experimental/SeparableLowPassImageFilter.h"
#include "stir/VoxelsOnCartesianGrid.h"

/*
    Copyright (C) 2000- 2011, IRSL
    See STIR/LICENSE.txt for details
*/
START_NAMESPACE_STIR

template <typename elemT>
SeparableLowPassImageFilter<elemT>::
SeparableLowPassImageFilter()
:filter_coefficients(VectorWithOffset<float>(-1,1))
{
    set_defaults();
}

template <typename elemT>
VectorWithOffset<float>
SeparableLowPassImageFilter<elemT>::
get_filter_coefficients()
{
  return filter_coefficients;
}
  
template <typename elemT>
Succeeded
SeparableLowPassImageFilter<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{
   
 /* VectorWithOffset<elemT> filter_coeff(filter_coefficients.size());

  for ( int i =1; i <=filter_coefficients.size(); i++)
  {
   filter_coeff[i] =  static_cast<elemT>(filter_coefficients[i]); 
  }*/


 const VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);

  lowpass_filter = 
    SeparableLowPassArrayFilter<3,elemT>(filter_coefficients, z_trivial);
  
  return Succeeded::yes;
  
}


template <typename elemT>
void
SeparableLowPassImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{ 
   //assert(consistency_check(density) == Succeeded::yes);
  lowpass_filter(density);  

  /*cerr <<"Line profile out " << endl;
       for (int i = -26;i<=26;i++)
	 cerr << density[10][1][i] << "   ";
       
       cerr << endl;*/
}


template <typename elemT>
void
SeparableLowPassImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density, 
	  const DiscretisedDensity<3,elemT>& in_density) const
{
  //assert(consistency_check(in_density) == Succeeded::yes);
  lowpass_filter(out_density,in_density);

  /* cerr <<"Line profile " << endl;

       for (int i = -26;i<=26;i++)
	 cerr << out_density[10][0][i] << "   ";
       
       cerr << endl;*/
	 


}


template <typename elemT>
void
SeparableLowPassImageFilter<elemT>::
set_defaults()
{
  //for ( int i = 0;i<=filter_coefficients.get_length();i++)
    filter_coefficients.fill(0);
    z_trivial =1;

    
}

template <typename elemT>
void 
SeparableLowPassImageFilter<elemT>::
initialise_keymap()
{
  parser.add_start_key("Separable Lowpass Filter Parameters");
  parser.add_key("filter_coefficients", &filter_coefficients_for_parsing);
  parser.add_key("z_trivial", &z_trivial);
  parser.add_stop_key("END Separable Lowpass Filter Parameters");

}

template <typename elemT>
bool 
SeparableLowPassImageFilter<elemT>::
post_processing()
{
  const unsigned int size = filter_coefficients_for_parsing.size();
  const int min_index = -static_cast<int>(size/2);
  filter_coefficients.grow(min_index, min_index + size - 1);
  for (int i = min_index; i<= filter_coefficients.get_max_index(); ++i)
    filter_coefficients[i] = 
      static_cast<float>(filter_coefficients_for_parsing[i-min_index]);
  return false;
}



const char * const 
SeparableLowPassImageFilter<float>::registered_name =
  "Separable Lowpass Filter";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static SeparableLowPassImageFilter<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template SeparableLowPassImageFilter<float>;

END_NAMESPACE_STIR
