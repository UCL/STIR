//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declares class ProjectorByBinPairUsingProjMatrixByBin

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#ifndef __Tomo_recon_buildblock_ProjectorByBinPairUsingProjMatrixByBin_h_
#define __Tomo_recon_buildblock_ProjectorByBinPairUsingProjMatrixByBin_h_

#include "tomo/RegisteredParsingObject.h"
#include "recon_buildblock/ProjectorByBinPair.h"
#include "recon_buildblock/ProjMatrixByBin.h"

START_NAMESPACE_TOMO


/*!
  \ingroup recon_buildblock
  \brief A projector pair based on a single matrix
*/
class ProjectorByBinPairUsingProjMatrixByBin : 
  public RegisteredParsingObject<ProjectorByBinPairUsingProjMatrixByBin,
                                 ProjectorByBinPair> 
{ 
public:
  //! Name which will be used when parsing a ProjectorByBinPair object
  static const char * const registered_name; 

  //! Default constructor 
  ProjectorByBinPairUsingProjMatrixByBin();

  //! Constructor that sets the pair
  ProjectorByBinPairUsingProjMatrixByBin(const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr);

  //! Stores all necessary geometric info
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );


  ProjMatrixByBin const * 
    get_proj_matrix_ptr() const;

private:

  shared_ptr<ProjMatrixByBin> proj_matrix_ptr;
  void set_defaults();
  void initialise_keymap();
  bool post_processing();
};

END_NAMESPACE_TOMO


#endif // __Tomo_recon_buildblock_ProjectorByBinPairUsingProjMatrixByBin_h_
