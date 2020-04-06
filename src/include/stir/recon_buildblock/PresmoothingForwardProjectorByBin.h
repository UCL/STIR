/*!
  \file

  \brief Declaration of class stir::PresmoothingForwardProjectorByBin

  \author Kris Thielemans
  \author Richard Brown

*/
/*
    Copyright (C) 2002- 2007, Hammersmith Imanet
    Copyright (C) 2019, University College London

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
#ifndef __stir_recon_buildblock_PresmoothingForwardProjectorByBin__H__
#define __stir_recon_buildblock_PresmoothingForwardProjectorByBin__H__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/shared_ptr.h"


START_NAMESPACE_STIR

template <typename elemT> class Viewgram;
template <typename DataT> class DataProcessor;
/*!
  \brief A very preliminary class that first smooths the image, then forward projects.

  \warning. It assumes that the DataProcessor does not change
  the size of the image.
*/
class PresmoothingForwardProjectorByBin : 
  public 
    RegisteredParsingObject<PresmoothingForwardProjectorByBin,
                            ForwardProjectorByBin>
{
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
public:
#else
 private:
#endif
  typedef ForwardProjectorByBin base_type;
public:
  //! Name which will be used when parsing a PresmoothingForwardProjectorByBin object
  static const char * const registered_name; 

  //! Default constructor (calls set_defaults())
  PresmoothingForwardProjectorByBin();

  ~ PresmoothingForwardProjectorByBin();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(           
                      const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<const DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );


  PresmoothingForwardProjectorByBin(
                                    const shared_ptr<ForwardProjectorByBin>& original_forward_projector_ptr,
                                    const shared_ptr<DataProcessor<DiscretisedDensity<3,float> > >&);

  // Informs on which symmetries the projector handles
  // It should get data related by at least those symmetries.
  // Otherwise, a run-time error will occur (unless the derived
  // class has other behaviour).
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;

private:

  shared_ptr<ForwardProjectorByBin> original_forward_projector_ptr;

#ifdef STIR_PROJECTORS_AS_V3
  void actual_forward_project(RelatedViewgrams<float>&, 
                              const DiscretisedDensity<3,float>&,
                              const int min_axial_pos_num, const int max_axial_pos_num,
                              const int min_tangential_pos_num, const int max_tangential_pos_num);
#endif
  void actual_forward_project(RelatedViewgrams<float>&,
                              const int min_axial_pos_num, const int max_axial_pos_num,
                              const int min_tangential_pos_num, const int max_tangential_pos_num);


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
