//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup projection

  \brief Declares class ProjectorByBinPair

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#ifndef __stir_recon_buildblock_ProjectorByBinPair_h_
#define __stir_recon_buildblock_ProjectorByBinPair_h_

#include "stir/RegisteredObject.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, class elemT> class DiscretisedDensity;
class ProjDataInfo;


/*!
  \ingroup projection
  \brief Abstract base class for all projector pairs

  This class is useful for all algorithms which need both a forward 
  and back projector. It's only purpose in that case is to provide the 
  parsing mechanisms, such that the projectors can be defined in a .par 
  file.
*/
class ProjectorByBinPair : 
  public RegisteredObject<ProjectorByBinPair> 
{ 
public:

  //! Default constructor 
  ProjectorByBinPair();

  virtual ~ProjectorByBinPair() {}

  //! Stores all necessary geometric info
  /*! 
  If necessary, set_up() can be called more than once.

  Derived classes can assume that the projectors  will be called
  with input corresponding to the arguments of the last call to set_up(). 
  */
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );


  //ForwardProjectorByBin const * 
  const shared_ptr<ForwardProjectorByBin>
    get_forward_projector_sptr() const;

  //BackProjectorByBin const *
  const shared_ptr<BackProjectorByBin>
    get_back_projector_sptr() const;
  

  //! Provide access to the (minimal) symmetries used by the projectors
  /*! It is expected that the forward and back projector can handle the same
      symmetries.

      \warning There is currently no check that this is the case, and we just return
      the symmetries returned by the back projector.
      \todo determine set of minimal symmetries
  */
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const
    {
      return get_back_projector_sptr()->get_symmetries_used();
    }

protected:

  shared_ptr<ForwardProjectorByBin> forward_projector_ptr;
  shared_ptr<BackProjectorByBin> back_projector_ptr;

};

END_NAMESPACE_STIR


#endif // __stir_recon_buildblock_ProjectorByBinPair_h_
