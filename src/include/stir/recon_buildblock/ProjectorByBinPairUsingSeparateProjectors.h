//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declares class ProjectorByBinPairUsingSeparateProjectors

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#ifndef __Tomo_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_
#define __Tomo_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_

#include "tomo/RegisteredParsingObject.h"
#include "recon_buildblock/ProjectorByBinPair.h"

START_NAMESPACE_TOMO


/*!
  \ingroup recon_buildblock
  \brief A projector pair based on a single matrix
*/
class ProjectorByBinPairUsingSeparateProjectors : 
  public RegisteredParsingObject<ProjectorByBinPairUsingSeparateProjectors,
                                 ProjectorByBinPair> 
{ 
public:
  //! Name which will be used when parsing a ProjectorByBinPair object
  static const char * const registered_name; 

  //! Default constructor 
  ProjectorByBinPairUsingSeparateProjectors();

   //! Constructor that sets the pair
  ProjectorByBinPairUsingSeparateProjectors(const shared_ptr<ForwardProjectorByBin>& forward_projector_ptr,
                                            const shared_ptr<BackProjectorByBin>& back_projector_ptr);


private:

  void set_defaults();
  void initialise_keymap();
};

END_NAMESPACE_TOMO


#endif // __Tomo_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_
