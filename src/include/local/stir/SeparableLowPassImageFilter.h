//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Declaration of class SeparableLowPassImageFilter
    
  \author Sanida Mustafovic
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SeparableLowPassImageFilter_H__
#define __stir_SeparableLowPassImageFilter_H__


#include "local/stir/SeparableLowPassArrayFilter.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"


START_NAMESPACE_STIR

// TODO!! remove define

#define num_dimensions 3

template <typename elemT>
class SeparableLowPassImageFilter : 
  public 
    RegisteredParsingObject<
        SeparableLowPassImageFilter<elemT>,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >
    >
{
 private:
  typedef 
    RegisteredParsingObject<
        SeparableLowPassImageFilter<elemT>,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >
    >
    base_type;
public:
  static const char * const registered_name; 
  
  //! Default constructor
  SeparableLowPassImageFilter();

  VectorWithOffset<float> get_filter_coefficients();
  
  
private:
  vector<double> filter_coefficients_for_parsing;
  VectorWithOffset<float> filter_coefficients;
  int z_trivial ;
  
   
  
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

END_NAMESPACE_STIR

#endif


