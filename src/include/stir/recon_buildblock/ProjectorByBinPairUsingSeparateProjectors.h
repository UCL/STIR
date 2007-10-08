//
// $Id$
//
/*!
  \file
  \ingroup projection

  \brief Declares class ProjectorByBinPairUsingSeparateProjectors

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_
#define __stir_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"

START_NAMESPACE_STIR


/*!
  \ingroup projection
  \brief A projector pair based on a single matrix
*/
class ProjectorByBinPairUsingSeparateProjectors : 
  public RegisteredParsingObject<ProjectorByBinPairUsingSeparateProjectors,
                                 ProjectorByBinPair,
                                 ProjectorByBinPair> 
{ 
 private:
  typedef
    RegisteredParsingObject<ProjectorByBinPairUsingSeparateProjectors,
                            ProjectorByBinPair,
                            ProjectorByBinPair> 
    base_type;
public:
  //! Name which will be used when parsing a ProjectorByBinPair object
  static const char * const registered_name; 

  //! Default constructor 
  ProjectorByBinPairUsingSeparateProjectors();

   //! Constructor that sets the pair
  ProjectorByBinPairUsingSeparateProjectors(const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
                                            const shared_ptr<BackProjectorByBin>& back_projector_sptr);


private:

  void set_defaults();
  void initialise_keymap();
  bool post_processing();
};

END_NAMESPACE_STIR


#endif // __stir_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_
