//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ImageProcessor

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_ImageProcessor_H__
#define __Tomo_ImageProcessor_H__


#include "tomo/RegisteredObject.h"

START_NAMESPACE_TOMO

template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup buildblock
  \brief 
  Base class that defines an interface for classes that do image processing.

  Leaves from ImageProcessor have to be able to parse parameter files etc. Moreover,
  it has to be possible to construct an ImageProcessor object while parsing, where
  the actual type of derived class is determined at run-time. 
  Luckily, this is exactly the situation that RegisteredObject and RegisteredParsingObject
  are supposed to handle.

 */
template <int num_dimensions, typename elemT>
class ImageProcessor : public RegisteredObject<ImageProcessor <num_dimensions,elemT> >
{
public:
  inline ImageProcessor();

  //! Builds the filter (if necessary) using \a density as a template for sizes, sampling distances etc.
  /*! 
     \warning Most derived classes will \b not rebuild the filter if the input data does
     not correspond to the template.
   */
  inline Succeeded  build_filter(const DiscretisedDensity< num_dimensions,elemT>& density);

  //! Builds the filter (if not already done before) and filter \a density in-place
  inline void apply(DiscretisedDensity<num_dimensions,elemT>& density);
  /*!
    \brief
    Builds the filter (if not already done before) and filter \a in_density, 
    putting the result in \a out_density.

    \warning Most derived classes will assume that out_density is already set-up
    appropriately.
  */
  inline void apply(DiscretisedDensity<num_dimensions,elemT>& out_density,
                               const DiscretisedDensity<num_dimensions,elemT>& in_density);

  // Check if filtering images with this dimensions, sampling_distances etc actually makes sense
  //virtual inline Succeeded consistency_check( const DiscretisedDensity<num_dimensions,elemT>& image ) const;  


protected:
  //! Will be called to build the filter
  virtual Succeeded  virtual_build_filter(const DiscretisedDensity< num_dimensions,elemT>& image) = 0;
  //! Performs actual filtering
  virtual void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density, 
                         const DiscretisedDensity<num_dimensions,elemT>& in_density) const = 0; 
  //! Performs actual filtering (in-place)
  virtual void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const =0;
private:  
  bool filter_built;  

};



END_NAMESPACE_TOMO

#include "tomo/ImageProcessor.inl"

#endif
