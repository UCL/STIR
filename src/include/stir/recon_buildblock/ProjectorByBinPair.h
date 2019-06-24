//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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

  \brief Declares class stir::ProjectorByBinPair

  \author Kris Thielemans

*/
#ifndef __stir_recon_buildblock_ProjectorByBinPair_h_
#define __stir_recon_buildblock_ProjectorByBinPair_h_

#include "stir/RegisteredObject.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/ParsingObject.h"
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
public RegisteredObject<ProjectorByBinPair> ,
public ParsingObject
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
  \warning Derived classes have to call set_up from the base class.
  */
  virtual Succeeded
    set_up(		 
	   const shared_ptr<ProjDataInfo>&,
	   const shared_ptr<DiscretisedDensity<3,float> >& // TODO should be Info only
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

  shared_ptr<ForwardProjectorByBin> forward_projector_sptr;
  shared_ptr<BackProjectorByBin> back_projector_sptr;

  //! check if the argument is the same as what was used for set_up()
  /*! calls error() if anything is wrong.

      If overriding this function in a derived class, you need to call this one.
   */
  virtual void check(const ProjDataInfo& proj_data_info, const DiscretisedDensity<3,float>& density_info) const;
  bool _already_set_up;

 private:
  shared_ptr<ProjDataInfo> _proj_data_info_sptr;
  //! The density ptr set with set_up()
  /*! \todo it is wasteful to have to store the whole image as this uses memory that we don't need. */
  shared_ptr<DiscretisedDensity<3,float> > _density_info_sptr;
};

END_NAMESPACE_STIR


#endif // __stir_recon_buildblock_ProjectorByBinPair_h_
