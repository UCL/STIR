//
// $Id$
//
/*!

  \file
  \ingroup ImageProcessor  
  \brief Declaration of class ThresholdMinToSmallPositiveValueImageProcessor
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ThresholdMinToSmallPositiveValueImageProcessor_H__
#define __stir_ThresholdMinToSmallPositiveValueImageProcessor_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/ImageProcessor.h"


START_NAMESPACE_STIR

// TODO!! remove define

#define num_dimensions 3

/*!
  \ingroup ImageProcessor  
  \brief A class in the ImageProcessor hierarchy for making sure all elements are strictly positive.

  Works by calling threshold_min_to_small_positive_value().
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.

  The discretised densities that will be filtered are supposed to be on a 
  Cartesian grid. This is for the rim_truncation part.

  \warning This class is currently restricted to 3d. This is mainly because of
  the difficulty to do the rim_truncation
  in n dimensions.
  \todo Remove the rim_truncation stuff. If necessary, this could be moved to
  a separate ImageProcessor
 */

template <typename elemT>
class ThresholdMinToSmallPositiveValueImageProcessor : 
  public 
    RegisteredParsingObject<
        ThresholdMinToSmallPositiveValueImageProcessor<elemT>,
        ImageProcessor<3,elemT>,
        ImageProcessor<3,elemT>
    >
{
public:
  static const char * const registered_name; 
  
  //! Construct given parameters 
  ThresholdMinToSmallPositiveValueImageProcessor(const int rim_truncation_image = 0);
    
  
private:
  
  int rim_truncation_image;
  
  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& image);

  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};

#undef num_dimensions

END_NAMESPACE_STIR

#endif


