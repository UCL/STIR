//
// $Id$: $Date$
//


#ifndef _BackProjectorByBinUsingProjMatrixByBin_
#define _BackProjectorByBinUsingProjMatrixByBin_

/*!

  \file

  \brief 

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "recon_buildblock/ProjMatrixByBin.h"
#include "recon_buildblock/BackProjectorByBin.h"
#include "shared_ptr.h"
#include "DataSymmetriesForBins.h"
//#include "RelatedViewgrams.h"

class Viewgrams;
template <typename elemT> class RelatedViewgrams;
class ProjDataInfoCylindricalArcCorr;


START_NAMESPACE_TOMO

/*!
  \brief This implements the BackProjectorByBin interface, given any 
ProjMatrixByBin object
    
  */
class BackProjectorByBinUsingProjMatrixByBin: public  BackProjectorByBin
{ 
public:
  BackProjectorByBinUsingProjMatrixByBin (  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr);
	 
	 
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;


  virtual void actual_back_project(DiscretisedDensity<3,float>& image,
                                   const RelatedViewgrams<float>&,
		                   const int min_axial_pos_num, const int max_axial_pos_num,
		                   const int min_tangential_pos_num, const int max_tangential_pos_num);


  shared_ptr<ProjMatrixByBin> &
    get_proj_matrix_sptr(){ return proj_matrix_ptr ;} 
  
  
protected:

  shared_ptr<ProjMatrixByBin> proj_matrix_ptr;
  
};


 

END_NAMESPACE_TOMO

//#include "recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.inl"

#endif


