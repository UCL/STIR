//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
  \ingroup recon_buildblock

  \brief Declaration of class stir::ecat::ecat7::BinNormalisationFromECAT7

  \author Kris Thielemans
*/

#ifndef HAVE_LLN_MATRIX
#error This file can only be compiled when HAVE_LLN_MATRIX is #defined
#endif

#ifndef __stir_recon_buildblock_BinNormalisationFromECAT7_H__
#define __stir_recon_buildblock_BinNormalisationFromECAT7_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/data/SinglesRates.h"
#include "stir/Scanner.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Array.h"
#include <string>

#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

/*!
  \ingroup recon_buildblock
  \ingroup ECAT
  \brief A BinNormalisation class that gets the normalisation factors from
  an ECAT7 3D normalisation file

  \par Parsing example
  \verbatim
  Bin Normalisation type := from ecat7
  Bin Normalisation From ECAT7:=
  normalisation filename:= myfile.n
  ; next keyword gives the singles to be used for dead-time correction
  ; normally set to get the info from the ECAT7 sinogram
  ;
  singles rates:= Singles From ECAT7
    Singles Rates From ECAT7 :=
       ecat7_filename := ${ecat7_filename}
     End Singles Rates From ECAT7:=

  ; next keywords can be used to switch off some of the normalisation components
  ; do not use unless you know why
  ; use_detector_efficiencies:=1
  ; use_dead_time:=1
  ; use_geometric_factors:=1
  ; use_crystal_interference_factors:=1
  End Bin Normalisation From ECAT7:=
  \endverbatim
 
*/
class BinNormalisationFromECAT7 :
   public RegisteredParsingObject<BinNormalisationFromECAT7, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char * const registered_name; 
  
  //! Default constructor
  /*! 
    \warning You should not call any member functions for any object just 
    constructed with this constructor. Initialise the object properly first
    by parsing.
  */
  BinNormalisationFromECAT7();

  //! Constructor that reads the projdata from a file
  BinNormalisationFromECAT7(const string& filename);

  virtual Succeeded set_up(const shared_ptr<ProjDataInfo>&);
  float get_bin_efficiency(const Bin& bin, const double start_time, const double end_time) const;

  bool use_detector_efficiencies() const;
  bool use_dead_time() const;
  bool use_geometric_factors() const;
  bool use_crystal_interference_factors() const;

private:
  Array<1,float> axial_t1_array;
  Array<1,float> axial_t2_array;
  Array<1,float> trans_t1_array;
  shared_ptr<SinglesRates> singles_rates_ptr;
  Array<2,float> geometric_factors;
  Array<2,float> efficiency_factors;
  Array<2,float> crystal_interference_factors;
  shared_ptr<Scanner> scanner_ptr;
  int num_transaxial_crystals_per_block;
  // TODO move to Scanner
  int num_axial_blocks_per_singles_unit;
  shared_ptr<ProjDataInfo> proj_data_info_ptr;
  ProjDataInfoCylindricalNoArcCorr const * proj_data_info_cyl_ptr;
  shared_ptr<ProjDataInfoCylindricalNoArcCorr> proj_data_info_cyl_uncompressed_ptr;
  int span;
  int mash;
  int num_blocks_per_singles_unit;

  bool _use_detector_efficiencies;
  bool _use_dead_time;
  bool _use_geometric_factors;
  bool _use_crystal_interference_factors;

  void read_norm_data(const string& filename);
  float get_dead_time_efficiency ( const DetectionPosition<>& det_pos,
				  const double start_time, const double end_time) const;

  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  string normalisation_ECAT7_filename;
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
