//
// $Id$
//
/*!

  \file

  \brief Declaration of class PostsmoothingBackProjectorByBin

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_PostsmoothingBackProjectorByBin__H__
#define __stir_recon_buildblock_PostsmoothingBackProjectorByBin__H__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/shared_ptr.h"


START_NAMESPACE_STIR

template <typename elemT> class Viewgram;
template <typename DataT> class DataProcessor;
/*!
  \brief A very preliminary class that first smooths the image, then back projects.

*/
class PostsmoothingBackProjectorByBin : 
  public 
    RegisteredParsingObject<PostsmoothingBackProjectorByBin,
                            BackProjectorByBin>
{
public:
  //! Name which will be used when parsing a PostsmoothingBackProjectorByBin object
  static const char * const registered_name; 

  //! Default constructor (calls set_defaults())
  PostsmoothingBackProjectorByBin();

  ~ PostsmoothingBackProjectorByBin();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
		      const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );


  PostsmoothingBackProjectorByBin(
                       const shared_ptr<BackProjectorByBin>& original_back_projector_ptr,
		       const shared_ptr<DataProcessor<DiscretisedDensity<3,float> > >&);

  // Informs on which symmetries the projector handles
  // It should get data related by at least those symmetries.
  // Otherwise, a run-time error will occur (unless the derived
  // class has other behaviour).
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;


private:

  shared_ptr<BackProjectorByBin> original_back_projector_ptr;
  shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > image_processor_ptr;

  void actual_back_project(DiscretisedDensity<3,float>&,
			   const RelatedViewgrams<float>&,
			   const int min_axial_pos_num, const int max_axial_pos_num,
			   const int min_tangential_pos_num, const int max_tangential_pos_num);



  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
