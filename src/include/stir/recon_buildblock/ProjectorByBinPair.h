//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declares class ProjectorByBinPair

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
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
  \ingroup recon_buildblock
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
  

protected:

  shared_ptr<ForwardProjectorByBin> forward_projector_ptr;
  shared_ptr<BackProjectorByBin> back_projector_ptr;

};

END_NAMESPACE_STIR


#endif // __stir_recon_buildblock_ProjectorByBinPair_h_
