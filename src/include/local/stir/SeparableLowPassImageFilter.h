//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Declaration of class SeparableLowPassImageFilter
    
  \author Sanida Mustafovic
  \author Kris Thielemans
      
  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_SeparableLowPassImageFilter_H__
#define __Tomo_SeparableLowPassImageFilter_H__


#include "local/tomo/SeparableLowPassArrayFilter.h"
#include "tomo/RegisteredParsingObject.h"
#include "tomo/ImageProcessor.h"


START_NAMESPACE_TOMO

// TODO!! remove define

#define num_dimensions 3

template <typename elemT>
class SeparableLowPassImageFilter : 
  public 
    RegisteredParsingObject<
        SeparableLowPassImageFilter<elemT>,
        ImageProcessor<num_dimensions,elemT>
    >
{
public:
  static const char * const registered_name; 
  
  //! Default constructor
  SeparableLowPassImageFilter();

  VectorWithOffset<float> get_filter_coefficients();
  
  
private:
  vector<double> filter_coefficients_for_parsing;
  VectorWithOffset<float> filter_coefficients;
  
   
  
  SeparableLowPassArrayFilter<num_dimensions,elemT> lowpass_filter;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  
  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& image);
  // new
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};

#undef num_dimensions

END_NAMESPACE_TOMO

#endif


