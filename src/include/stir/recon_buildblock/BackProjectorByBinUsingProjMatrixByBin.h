//
// $Id$
//


#ifndef _BackProjectorByBinUsingProjMatrixByBin_
#define _BackProjectorByBinUsingProjMatrixByBin_

/*!

  \file

  \brief Declaration of class stir::BackprojectorByBinUsingProjMatrixByBin

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
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

#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
//#include "stir/DataSymmetriesForBins.h"
//#include "stir/RelatedViewgrams.h"

class Viewgrams;
template <typename elemT> class RelatedViewgrams;
class ProjDataInfoCylindricalArcCorr;


START_NAMESPACE_STIR

/*!
  \brief This implements the BackProjectorByBin interface, given any 
ProjMatrixByBin object
    
  */
class BackProjectorByBinUsingProjMatrixByBin: 
  public RegisteredParsingObject<BackProjectorByBinUsingProjMatrixByBin,
                                 BackProjectorByBin>
{ 
public:
    //! Name which will be used when parsing a BackProjectorByBin object
  static const char * const registered_name; 

  BackProjectorByBinUsingProjMatrixByBin();

  BackProjectorByBinUsingProjMatrixByBin (  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr);

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );
	 
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;


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
  virtual bool post_processing();

};


 

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.inl"

#endif


