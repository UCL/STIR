//
// $Id$
//

#ifndef __ImageFilter_h_
#define __ImageFilter_h_

/*!
  \file 
  \ingroup buildblock
 
  \brief declares the ImageFilter class

  \author Matthew Jacobson (some help by Kris Thielemans)
  \author based on C-code by Roni Levkowitz
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/

#include "VectorWithOffset.h"



START_NAMESPACE_TOMO

/*!
 \brief Class for filters that are applied to image objects

Currently, only a separable Metz filter appropriate for Cartesian images has been implemented.

*/

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class ImageFilter
{

 public:

  //! generates the filter kernels
  void build(const DiscretisedDensity<3,float>& representative_image,double fwhmx_dir=0,double fwhmz_dir=0,float Nx_dir=0.0,float Nz_dir=0.0);

  //! does 3D convolution with the filter
  void apply(DiscretisedDensity<3,float>& input_image, bool applying_threshold=false);

  //! constructor
   ImageFilter();

  //! destructor
  ~ImageFilter();

  //! used to see if kernels have been generated
  bool kernels_built;

 private:
   
  //! kernel lengths
  int kerlx,kerlz;

  //! arrays of kernel data and to convolution input/output
  VectorWithOffset<float> onerow,outrow,kernelX,kernelZ;

  //! full width at half maximum parameters
  double fwhmx,fwhmz;

  //! Metz powers
  float Nx,Nz;  //MJ 17/12/98 added Metz parameters

};

END_NAMESPACE_TOMO

#endif // __ImageFilter_h_

