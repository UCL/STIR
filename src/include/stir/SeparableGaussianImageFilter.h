/*!

  \file
  \ingroup buildblock  
  \brief Declaration of class stir::SeparableGaussianImageFilter
    
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Ludovica Brusaferri
      
*/
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet
    Copyright (C) 2018, UCL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SeparableGaussianImageFilter_H__
#define __stir_SeparableGaussianImageFilter_H__


#include "stir/SeparableGaussianArrayFilter.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"


START_NAMESPACE_STIR

// TODO!! remove define

#define num_dimensions 3

/*!
  \ingroup ImageProcessor
  \brief A class in the DataProcessor hierarchy that implements Gaussian filtering.

  As it is derived from RegisteredParsingObject, it implements all the
  necessary things to parse parameter files etc.

  The discretised densities that will be filtered are supposed to be on a
  Cartesian grid. The filtering operation is then performed as 3 separate
  1d filters in every direction.

  \warning This class is currently restricted to 3d. This is mainly because of
  the difficulty to give sensible names to the parameters used for parsing
  in n dimensions.

  \see SeparableGaussianArrayFilter for what a Gaussian filter is
 */

template <typename elemT>
class SeparableGaussianImageFilter : 
  public 
    RegisteredParsingObject<
        SeparableGaussianImageFilter<elemT>,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >
    >
{
 private:
  typedef
    RegisteredParsingObject<
        SeparableGaussianImageFilter<elemT>,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >,
        DataProcessor<DiscretisedDensity<num_dimensions,elemT> >
    >
    base_type;
public:
  static const char * const registered_name; 
  
  //! Default constructor
  SeparableGaussianImageFilter();

  BasicCoordinate< num_dimensions,float> get_fwhms();
  BasicCoordinate< num_dimensions,int> get_max_kernel_sizes();
  bool get_normalised_filter();
  
    void set_fwhms(const BasicCoordinate< num_dimensions,float>&);
    void set_max_kernel_sizes(const BasicCoordinate< num_dimensions,int>&);
    void set_normalise(const bool);
    
private:
  BasicCoordinate< num_dimensions,float> fwhms;

protected:

  BasicCoordinate< num_dimensions,int> max_kernel_sizes;
  bool normalise;
  
  SeparableGaussianArrayFilter<num_dimensions,elemT> gaussian_filter;

  virtual void set_defaults();
  virtual void initialise_keymap();

  //virtual bool post_processing();
  
  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& image);
  // new
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const ;
};

#undef num_dimensions

END_NAMESPACE_STIR

#endif


