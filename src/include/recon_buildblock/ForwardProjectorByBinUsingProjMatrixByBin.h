//
// $Id$
//
#ifndef _ForwardProjectorByBinUsingProjMatrixByBin_
#define _ForwardProjectorByBinUsingProjMatrixByBin_

/*!

  \file
  \ingroup recon_buildblock
  
  \brief definition of ForwardProjectorByBinUsingProjMatrixByBin
    
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Mustapha Sadki
  \author PARAPET project
      
  $Date$       
  $Revision$
*/


#include "recon_buildblock/ProjMatrixByBin.h"
#include "recon_buildblock/ForwardProjectorByBin.h"
#include "shared_ptr.h"
#include "DataSymmetriesForBins.h"


START_NAMESPACE_TOMO

template <typename elemT> class RelatedViewgrams;

/*!
  \brief This implements the ForwardProjectorByBin interface, given any 
  ProjMatrixByBin object
  \ingroup recon_buildblock

  It stores a shared_ptr to a ProjMatrixByBin object, which will be used
  to get the relevant elements of the projection matrix.
  */
class ForwardProjectorByBinUsingProjMatrixByBin: public  ForwardProjectorByBin
{ 
public:
  
  ForwardProjectorByBinUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr
    );
    
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;

  
private:
  shared_ptr<ProjMatrixByBin>  proj_matrix_ptr;
  

  void actual_forward_project(RelatedViewgrams<float>&, 
			      const DiscretisedDensity<3,float>& image,
			      const int min_axial_pos_num, const int max_axial_pos_num,
			      const int min_tangential_pos_num, const int max_tangential_pos_num);
  
};


END_NAMESPACE_TOMO


#endif


