//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Declaration of class ChainedImageProcessor
    
  \author Kris Thielemans
      
  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_ChainedImageProcessor_H__
#define __Tomo_ChainedImageProcessor_H__


#include "tomo/RegisteredParsingObject.h"
#include "tomo/ImageProcessor.h"


START_NAMESPACE_TOMO


/*!
  \brief A class in the ImageProcessor hierarchy that calls
   2 ImageProcessors in sequence.
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.
  \warning The 2 argument version of  ChainedImageProcessor::build_and_filter
  calls the first image processor with a temporary output density with 
  the same characteristics as the input density. 
  

  \warning ChainedImageProcessor::build_filter builds only the image
  processor of the first in the image processor in the chain. This 
  is because at this point, we do not really know what the first
  image processor will do to the image (it might change index 
  ranges or so), so it is impossible to build the 2nd image 
  processor without actually letting the first image processor 
  process the iamge (which might be far too expensive). This is not a real
  problem however, as  ChainedImageProcessor::build_and_filter is
  fine.
 */

template <int num_dimensions, typename elemT>
class ChainedImageProcessor : 
  public 
    RegisteredParsingObject<
        ChainedImageProcessor<num_dimensions,elemT>,
        ImageProcessor<num_dimensions,elemT>
    >
{
public:
  static const char * const registered_name; 
  
  //! Construct given parameters 
  /*! \warning parameter type will be changed to shared_ptr
   */
  explicit
  ChainedImageProcessor(ImageProcessor<num_dimensions,elemT> *const apply_first=0,
			ImageProcessor<num_dimensions,elemT> *const apply_second=0);
    
  
private:
  
  ImageProcessor<num_dimensions,elemT> * apply_first;
  ImageProcessor<num_dimensions,elemT> * apply_second;
  
  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_build_filter(const DiscretisedDensity<num_dimensions,elemT>& image);

  void  filter_it(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  filter_it(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};


END_NAMESPACE_TOMO

#endif


