//
// $Id: BackProjectorByBinUsingSquareProjMatrixByBin.h
//


#ifndef _BackProjectorByBinUsingSquareProjMatrixByBin_
#define _BackProjectorByBinUsingSquareProjMatrixByBin_

/*!

  \file

  \brief 

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

  $Date: 

  $Revision: 
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
//#include "stir/recon_buildblock/DataSymmetriesForBins.h"
//#include "stir/RelatedViewgrams.h"

class Viewgrams;
template <typename elemT> class RelatedViewgrams;
class ProjDataInfoCylindricalArcCorr;


START_NAMESPACE_STIR

/*!
  \brief This implements the BackProjectorByBin interface, given any 
ProjMatrixByBin object
    
  */
class BackProjectorByBinUsingSquareProjMatrixByBin: 
 public RegisteredParsingObject<BackProjectorByBinUsingSquareProjMatrixByBin,
                                 BackProjectorByBin>
{ 
public:    
  static const char * const registered_name; 

  BackProjectorByBinUsingSquareProjMatrixByBin();

  BackProjectorByBinUsingSquareProjMatrixByBin (  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr);
	 
	 
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;

   virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );


  virtual void actual_back_project(DiscretisedDensity<3,float>& image,
                                   const RelatedViewgrams<float>&,
		                   const int min_axial_pos_num, const int max_axial_pos_num,
		                   const int min_tangential_pos_num, const int max_tangential_pos_num);


  shared_ptr<ProjMatrixByBin> &
    get_proj_matrix_sptr(){ return proj_matrix_ptr ;} 
  
 protected:

  shared_ptr<ProjMatrixByBin> proj_matrix_ptr;

private:
  virtual void set_defaults();
  virtual void initialise_keymap();
};


 

END_NAMESPACE_STIR

//#include "stir/BackProjectorByBinUsingSquareProjMatrixByBin.inl"

#endif


