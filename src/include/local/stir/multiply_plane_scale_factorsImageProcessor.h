//
//
/*!

  \file
  \ingroup ImageProcessor  
  \brief Declaration of class multiply_plane_scale_factorsImageProcessor
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_multiply_plane_scale_factorsImageProcessor_H__
#define __stir_multiply_plane_scale_factorsImageProcessor_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::vector;
#endif

START_NAMESPACE_STIR

template <typename elemT> class VectorWithOffset;

/*!
  \brief Simply multiplies each plane in an image with a scale factor.
 */

template <typename elemT>
class multiply_plane_scale_factorsImageProcessor : 
  public 
    RegisteredParsingObject<
        multiply_plane_scale_factorsImageProcessor<elemT>,
        DataProcessor<DiscretisedDensity<3,elemT> >,
        DataProcessor<DiscretisedDensity<3,elemT> >
    >
{
private:
  typedef
    RegisteredParsingObject<
        multiply_plane_scale_factorsImageProcessor<elemT>,
        DataProcessor<DiscretisedDensity<3,elemT> >,
        DataProcessor<DiscretisedDensity<3,elemT> >
    >
    base_type;
public:
  static const char * const registered_name;   
  multiply_plane_scale_factorsImageProcessor();
  multiply_plane_scale_factorsImageProcessor(const VectorWithOffset<double>&  plane_scale_factors);
  multiply_plane_scale_factorsImageProcessor(const vector<double>&  plane_scale_factors);
    
  
private:
  vector<double>  plane_scale_factors;

  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_set_up(const DiscretisedDensity<3,elemT>& image);

  void  virtual_apply(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<3,elemT>& density) const ;
  
};

END_NAMESPACE_STIR

#endif


