//
// $Id$
//

#ifndef __ImageFilter_h_
#define __ImageFilter_h_

/*!
  \file 
  \ingroup buildblock
 
  \brief declares the ImageFilter class

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/

#include "Tomography_common.h"



// MJ 02/03/2000 ftype ---> kernels_built

START_NAMESPACE_TOMO

/*!
 \brief Class for filters that are applied to image objects

Currently, only a separable Metz filter appropriate for Cartesian images has been implemented.

*/

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class ImageFilter
{

  //! kernel lengths
  long kerlx,kerlz;

  //! point to arrays of kernel data and to convolution input/output
  float *onerow,*outrow,*kernelX,*kernelZ;

  //! full width at half maximum parameters
  double fwhmx,fwhmz;

  //! Metz powers
  float Nx,Nz;  //MJ 17/12/98 added Metz parameters

 public:

 //MJ 09/02/99 Added Metz filter parameters as function arguments

 //MJ 05/03/2000 got rid of scanner dependence, changed defaults

  //! generates the filter kernels
  void build(const DiscretisedDensity<3,float>& representative_image,double fwhmx_dir=0,double fwhmz_dir=0,float Nx_dir=0.0,float Nz_dir=0.0);

  //! does 3D convolution with the 
  void apply(DiscretisedDensity<3,float>& input_image);

  //! constructor
   ImageFilter();

  //! destructor
  ~ImageFilter();

  //! used to see if kernels have been generated
  bool kernels_built;

};

END_NAMESPACE_TOMO

#endif // __ImageFilter_h_

