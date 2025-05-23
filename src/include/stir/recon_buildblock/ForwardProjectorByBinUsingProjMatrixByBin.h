//
//
#ifndef _ForwardProjectorByBinUsingProjMatrixByBin_
#define _ForwardProjectorByBinUsingProjMatrixByBin_

/*!

  \file
  \ingroup projection
  
  \brief definition of stir::ForwardProjectorByBinUsingProjMatrixByBin
    
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Mustapha Sadki
  \author PARAPET project
      
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"



START_NAMESPACE_STIR

template <typename elemT> class RelatedViewgrams;

/*!
  \brief This implements the ForwardProjectorByBin interface, given any 
  ProjMatrixByBin object
  \ingroup projection

  It stores a shared_ptr to a ProjMatrixByBin object, which will be used
  to get the relevant elements of the projection matrix.
  */
class ForwardProjectorByBinUsingProjMatrixByBin: 
  public RegisteredParsingObject<ForwardProjectorByBinUsingProjMatrixByBin,
                                 ForwardProjectorByBin>
{ 
public:
    //! Name which will be used when parsing a ForwardProjectorByBin object
  static const char * const registered_name; 

  ForwardProjectorByBinUsingProjMatrixByBin();

  ForwardProjectorByBinUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr
    );

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<const DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

  
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;

  
private:
  shared_ptr<ProjMatrixByBin>  proj_matrix_ptr;
  

  void actual_forward_project(RelatedViewgrams<float>&, 
			      const DiscretisedDensity<3,float>& image,
			      const int min_axial_pos_num, const int max_axial_pos_num,
			      const int min_tangential_pos_num, const int max_tangential_pos_num);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  
};


END_NAMESPACE_STIR


#endif


