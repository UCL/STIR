//
// $Id$
//
/*!

  \file
  \ingroup ImageProcessor  
  \brief Declaration of class ChainedImageProcessor
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ChainedImageProcessor_H__
#define __stir_ChainedImageProcessor_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/ImageProcessor.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR


/*!
  \ingroup ImageProcessor  
  \brief A class in the ImageProcessor hierarchy that calls
   2 ImageProcessors in sequence.
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.
  \warning The 2 argument version of  ChainedImageProcessor::apply
  calls the first image processor with a temporary output density with 
  the same characteristics as the input density. 
  

  \warning ChainedImageProcessor::set_up builds only the image
  processor of the first in the image processor chain. This 
  is because at this point, we do not really know what the first
  image processor will do to the image (it might change index 
  ranges or so), so it is impossible to build the 2nd image 
  processor without actually letting the first image processor 
  process the image (which might be far too expensive). This is not a real
  problem however, as  ChainedImageProcessor::apply is
  fine as it will  call virtual_set_up for the 2nd
  image processor anyway.
 */

template <int num_dimensions, typename elemT>
class ChainedImageProcessor : 
  public 
    RegisteredParsingObject<
        ChainedImageProcessor<num_dimensions,elemT>,
        ImageProcessor<num_dimensions,elemT>,
        ImageProcessor<num_dimensions,elemT>
    >
{
public:
  static const char * const registered_name; 
  
  //! Construct given ImageProcessor parameters
#ifndef _MSC_VER
  explicit
  ChainedImageProcessor(shared_ptr<ImageProcessor<num_dimensions,elemT> > const& apply_first=0,
			shared_ptr<ImageProcessor<num_dimensions,elemT> > const& apply_second=0);
#else
  // VC does not compile the above for some reason
  // but gcc does not compile the work-around below...
  explicit
  ChainedImageProcessor(shared_ptr<ImageProcessor<num_dimensions,elemT> > const& apply_first=
			  shared_ptr<ImageProcessor<num_dimensions,elemT> >(),
			shared_ptr<ImageProcessor<num_dimensions,elemT> > const& apply_second=
			  shared_ptr<ImageProcessor<num_dimensions,elemT> >());
#endif    
  
private:
  
  shared_ptr<ImageProcessor<num_dimensions,elemT> > apply_first;
  shared_ptr<ImageProcessor<num_dimensions,elemT> > apply_second;
  
  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& image);

  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};


END_NAMESPACE_STIR

#endif


