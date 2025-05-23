/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*! 
  \file
  \ingroup InterfileIO
  \brief  This file declares the class stir::InterfilePDFSHeaderSPECT

  \author Berta Marti Fuster
  \author Kris Thielemans
*/


#ifndef __stir_INTERFILEPDFSHEADERSPECT_H__
#define __stir_INTERFILEPDFSHEADERSPECT_H__

#include "stir/IO/InterfileHeader.h"


START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \brief a class for Interfile keywords (and parsing) specific to 
  SPECT projection data
  \ingroup InterfileIO
  */
class InterfilePDFSHeaderSPECT : public InterfileHeader
{
public:
  InterfilePDFSHeaderSPECT();

protected:

  //! Returns false if OK, true if not.
  virtual bool post_processing();

 private:

  //! in mm
  std::vector<double> radii_of_rotation;
  
  int num_views;
  int num_bins;
  int start_angle;
  std::string direction_of_rotation;
  double extent_of_rotation;
  std::string orbit;
public:  
  ProjDataFromStream::StorageOrder storage_order;
  shared_ptr<const ProjDataInfo> data_info_sptr;

private:
  int num_segments;
  // ! for circular orbits (in mm )
  double radius_of_rotation;

  double bin_size_in_cm; 
  int num_axial_poss;

};


END_NAMESPACE_STIR

#endif // __stir_INTERFILEHEADERSPECT_H__
