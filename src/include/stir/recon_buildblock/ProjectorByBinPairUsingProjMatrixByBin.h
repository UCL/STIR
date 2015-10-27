//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup projection

  \brief Declares class stir::ProjectorByBinPairUsingProjMatrixByBin

  \author Kris Thielemans

*/
#ifndef __stir_recon_buildblock_ProjectorByBinPairUsingProjMatrixByBin_h_
#define __stir_recon_buildblock_ProjectorByBinPairUsingProjMatrixByBin_h_

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"

START_NAMESPACE_STIR

class Succeeded;
/*!
  \ingroup projection
  \brief A projector pair based on a single matrix
*/
class ProjectorByBinPairUsingProjMatrixByBin : 
  public RegisteredParsingObject<ProjectorByBinPairUsingProjMatrixByBin,
                                 ProjectorByBinPair,
                                 ProjectorByBinPair> 
{ 
 private:
  typedef
    RegisteredParsingObject<ProjectorByBinPairUsingProjMatrixByBin,
                            ProjectorByBinPair,
                            ProjectorByBinPair> 
    base_type;
public:
  //! Name which will be used when parsing a ProjectorByBinPair object
  static const char * const registered_name; 

  //! Default constructor 
  ProjectorByBinPairUsingProjMatrixByBin();

  //! Constructor that sets the projection matrix
  ProjectorByBinPairUsingProjMatrixByBin(const shared_ptr<ProjMatrixByBin>& proj_matrix_sptr);

  //! Stores all necessary geometric info
  /*! First constructs forward and back projectors and then calls base_type::setup */
  virtual Succeeded set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr // TODO should be Info only
    );

  ProjMatrixByBin const * 
    get_proj_matrix_ptr() const;

  void set_proj_matrix_sptr(const shared_ptr<ProjMatrixByBin>& sptr);

private:

  shared_ptr<ProjMatrixByBin> proj_matrix_sptr;
  void set_defaults();
  void initialise_keymap();
  bool post_processing();
};

END_NAMESPACE_STIR


#endif // __stir_recon_buildblock_ProjectorByBinPairUsingProjMatrixByBin_h_
