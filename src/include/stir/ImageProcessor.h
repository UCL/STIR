//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ImageProcessor

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ImageProcessor_H__
#define __stir_ImageProcessor_H__


#include "stir/RegisteredObject.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup buildblock
  \brief 
  Base class that defines an interface for classes that do image processing.

  Classes at the end of the ImageProcessor hierarchy have to be able 
  to parse parameter files etc. Moreover,
  it has to be possible to construct an ImageProcessor object while 
  parsing, where the actual type of derived class is determined at 
  run-time. 
  Luckily, this is exactly the situation that RegisteredObject and 
  RegisteredParsingObject are supposed to handle. So, all this 
  functionality is achieved by deriving ImageProcessor from the
  appropriate RegisteredObject class, and deriving the 'leaves'
  from Registered ParsingObject.
 */
template <int num_dimensions, typename elemT>
class ImageProcessor : 
  public RegisteredObject<ImageProcessor <num_dimensions,elemT> >
{
public:
  inline ImageProcessor();

  //! Initialises any internal data (if necessary) using \a density as a template for sizes, sampling distances etc.
  /*! 
     \warning ImageProcessor does NOT check if the input data for apply()
     actually corresponds to the template. So, most derived classes will 
     \b not call set_up again if the input data does
     not correspond to the template, potentially resulting in erroneous output.

     The reason that ImageProcessor does not perform this check is that
     it does not know what the requirements are to call the 2 densities
     'compatible'.
   */
  inline Succeeded  
    set_up(const DiscretisedDensity< num_dimensions,elemT>& density);

  //! Calls set_up() (if not already done before) and process \a density in-place
  /*! If set_up() returns Succeeded::false, a warning message is written, 
      and the \a density is not changed.
  */
  inline void apply(DiscretisedDensity<num_dimensions,elemT>& density);

  /*!
    \brief
    Calls set_up() (if not already done before) and process \a in_density, 
    putting the result in \a out_density.

    If set_up() returns Succeeded::false, a warning message is written, 
    and the \a out_density is not changed.

    \warning Most derived classes will assume that out_density is already 
    initialised appropriately (i.e. has correct dimensions, voxel sizes etc.).
  */
  inline void apply(DiscretisedDensity<num_dimensions,elemT>& out_density,
		    const DiscretisedDensity<num_dimensions,elemT>& in_density);

  // Check if filtering images with this dimensions, sampling_distances etc actually makes sense
  //virtual inline Succeeded consistency_check( const DiscretisedDensity<num_dimensions,elemT>& image ) const;  


protected:
  //! Will be called to build any internal parameters
  virtual Succeeded  virtual_set_up(const DiscretisedDensity< num_dimensions,elemT>& image) = 0;
  //! Performs actual operation (virtual_set_up is called before this function)
  virtual void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density, 
                         const DiscretisedDensity<num_dimensions,elemT>& in_density) const = 0; 
  //! Performs actual operation (in-place)
  virtual void virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const =0;
private:  
  bool is_set_up_already;  

};



END_NAMESPACE_STIR

#include "stir/ImageProcessor.inl"

#endif
