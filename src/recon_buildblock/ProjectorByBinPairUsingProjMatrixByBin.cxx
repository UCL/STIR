//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjectorByBinPairUsingProjMatrixByBin
  
  \author Kris Thielemans
    
  $Date$
  $Revision$
*/


#include "recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"

START_NAMESPACE_TOMO


const char * const 
ProjectorByBinPairUsingProjMatrixByBin::registered_name =
  "Matrix";


void 
ProjectorByBinPairUsingProjMatrixByBin::initialise_keymap()
{
  parser.add_start_key("Projector Pair Using Matrix Parameters");
  parser.add_stop_key("End Projector Pair Using Matrix Parameters");
  parser.add_parsing_key("Matrix type",&proj_matrix_ptr);
}


void
ProjectorByBinPairUsingProjMatrixByBin::set_defaults()
{}

bool
ProjectorByBinPairUsingProjMatrixByBin::post_processing()
{
  if (proj_matrix_ptr.use_count()==0)
    { warning("No valid projection matrix is defined\n"); return true; }
  forward_projector_ptr = new ForwardProjectorByBinUsingProjMatrixByBin(proj_matrix_ptr);
  back_projector_ptr = new BackProjectorByBinUsingProjMatrixByBin(proj_matrix_ptr);
  return false;
}

ProjectorByBinPairUsingProjMatrixByBin::
ProjectorByBinPairUsingProjMatrixByBin()
{
  set_defaults();
}

ProjectorByBinPairUsingProjMatrixByBin::
ProjectorByBinPairUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr)	   
    : proj_matrix_ptr(proj_matrix_ptr)
{}

void
ProjectorByBinPairUsingProjMatrixByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)

{    	   
  proj_matrix_ptr->set_up(proj_data_info_ptr, image_info_ptr);
}

ProjMatrixByBin const * 
ProjectorByBinPairUsingProjMatrixByBin::
get_proj_matrix_ptr() const
{
  return proj_matrix_ptr.get();
}

END_NAMESPACE_TOMO
