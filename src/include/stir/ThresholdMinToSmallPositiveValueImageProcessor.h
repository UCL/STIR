//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Declaration of class TruncateMinToSmallPositiveValueImageProcessor
    
  \author Kris Thielemans
      
  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_TruncateMinToSmallPositiveValueImageProcessor_H__
#define __Tomo_TruncateMinToSmallPositiveValueImageProcessor_H__


#include "tomo/RegisteredParsingObject.h"
#include "tomo/ImageProcessor.h"


START_NAMESPACE_TOMO

// TODO!! remove define

#define num_dimensions 3

/*!
  \brief A class in the ImageProcessor hierarchy that calls
   truncate_min_to_small_positive_value().
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.

  The discretised densities that will be filtered are supposed to be on a 
  Cartesian grid. This is for the rim_truncation part.

  \warning This class is currently restricted to 3d. This is mainly because of
  the difficulty to do the rim_truncation
  in n dimensions.
 */

template <typename elemT>
class TruncateMinToSmallPositiveValueImageProcessor : 
  public 
    RegisteredParsingObject<
        TruncateMinToSmallPositiveValueImageProcessor<elemT>,
        ImageProcessor<num_dimensions,elemT>
    >
{
public:
  static const char * const registered_name; 
  
  //! Construct given parameters 
  TruncateMinToSmallPositiveValueImageProcessor(const int rim_truncation_image = 0);
    
  
private:
  
  int rim_truncation_image;
  
  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_build_filter(const DiscretisedDensity<num_dimensions,elemT>& image);

  void  filter_it(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  filter_it(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};

#undef num_dimensions

END_NAMESPACE_TOMO

#endif


