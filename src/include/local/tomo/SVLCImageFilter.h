#if 0
//
// %W%: %E%
//
/*!

  \file

  \brief 

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  \date %E%
  \version %I%
*/

#ifndef __SVLCImageFilter_H__
#define __SVLCImageFilter_H__


#include "tomo/ImageFilter.h"
#include "tomo/DAVArrayFilter3D.h"

#include "tomo/RegisteredParsingObject.h"

START_NAMESPACE_TOMO

template <typename coordT> class CartesianCoordinate3D;

template <typename elemT>
class SVLCImageFilter:
  public 
      RegisteredParsingObject<
	      SVLCImageFilter<elemT>,
              ImageFilter<3,elemT>
	       >

{
public:
  static const char * const registered_name; 

  SVLCImageFilter();

  SVLCImageFilter(const CartesianCoordinate3D<int>& mask_radius);  
 
private:
  SVLCArrayFilter<elemT> svlc_filter;
  int mask_radius_x;
  int mask_radius_y;
  int mask_radius_z;


  virtual void set_defaults();
  virtual void initialise_keymap();

  Succeeded virtual_build_filter (const DiscretisedDensity< 3,elemT>& density);
  void filter_it(DiscretisedDensity<3,elemT>& density, const DiscretisedDensity<3,elemT>& in_density) const; 
  void filter_it(DiscretisedDensity<3,elemT>& density) const; 
};


END_NAMESPACE_TOMO


#endif  // SVLCImageFilter


#endif