//
//
/*!
  \file

  \brief Declaration of class stir::PostsmoothingBackProjectorByBin

  \author Kris Thielemans

*/
/*
    Copyright (C) 2002- 2007, Hammersmith Imanet

    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.


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

  void update_filtered_density_image(DiscretisedDensity<3, float>&);

  void init_filtered_density_image(DiscretisedDensity<3, float> &);

  BackProjectorByBin* get_original_back_projector_ptr() const;

private:

  shared_ptr<BackProjectorByBin> original_back_projector_ptr;
  shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > image_processor_ptr;

  void actual_back_project(DiscretisedDensity<3,float>&,
                           const RelatedViewgrams<float>&,
                           const int min_axial_pos_num, const int max_axial_pos_num,
                           const int min_tangential_pos_num, const int max_tangential_pos_num);

  void actual_back_project(DiscretisedDensity<3,float>& density,
                                   const Bin& bin);

  shared_ptr<DiscretisedDensity<3,float> > filtered_density_sptr;


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
