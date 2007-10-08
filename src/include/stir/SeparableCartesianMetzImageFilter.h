//
// $Id$
//
/*!

  \file
  \ingroup ImageProcessor  
  \brief Declaration of class SeparableCartesianMetzImageFilter
    
  \author Sanida Mustafovic
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SeparableCartesianMetzImageFilter_H__
#define __stir_SeparableCartesianMetzImageFilter_H__


#include "stir/SeparableMetzArrayFilter.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"


START_NAMESPACE_STIR

// TODO!! remove define

#define num_dimensions 3

/*!
  \ingroup ImageProcessor  
  \brief A class in the ImageProcessor hierarchy that implements Metz 
  filtering (which includes Gaussian filtering).
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.

  The discretised densities that will be filtered are supposed to be on a 
  Cartesian grid. The filtering operation is then performed as 3 separate
  1d filters in every direction.

  \warning This class is currently restricted to 3d. This is mainly because of
  the difficulty to give sensible names to the parameters used for parsing
  in n dimensions.

  \see SeparableMetzArrayFilter for what a Metz filter is
 */

template <typename elemT>
class SeparableCartesianMetzImageFilter : 
  public 
    RegisteredParsingObject<
        SeparableCartesianMetzImageFilter<elemT>,
        DataProcessor<DiscretisedDensity<3,elemT> >,
        DataProcessor<DiscretisedDensity<3,elemT> >
    >
{
 private:
  typedef
    RegisteredParsingObject<
              SeparableCartesianMetzImageFilter<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >
    base_type;
public:
  static const char * const registered_name; 
  
  //! Default constructor
  SeparableCartesianMetzImageFilter();
  
  // Construct metz filter given parameters 
  //SeparableCartesianMetzImageFilter(const double fwhm_x,const double fwhm_y, const double fwhm_z,const int metz_power_x,const int metz_power_y, const int metz_power_z);
  
  //Succeeded consistency_check( const DiscretisedDensity<num_dimensions,elemT>& image) const;  
  
  
  VectorWithOffset<float> get_metz_fwhms() const;
  VectorWithOffset<float> get_metz_powers() const;
  //! Maximum number of elements in the kernels
  /*! -1 means unrestricted*/
  VectorWithOffset<int> get_max_kernel_sizes() const;  
  
private:
  
  VectorWithOffset<float> fwhms;
  VectorWithOffset<float> metz_powers;
  VectorWithOffset<int> max_kernel_sizes;  
  
  SeparableMetzArrayFilter<num_dimensions,elemT> metz_filter;

  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& image);
  // new
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};

#undef num_dimensions

END_NAMESPACE_STIR

#endif


