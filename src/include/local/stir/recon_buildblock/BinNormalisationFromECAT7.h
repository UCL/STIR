//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class BinNormalisationFromECAT7

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
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
#include "local/stir/SinglesRates.h"
#include "stir/Scanner.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Array.h"
#include <string>

#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT7

/*!
  \ingroup recon_buildblock
  \ingroup ECAT
  \brief A BinNormalisation class that gets the normalisation factors from
  an ECAT7 3D normalisation file
 
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
  float get_bin_efficiency(const Bin& bin) const;
  //! Normalise some data
  /*! 
    This means \c multiply with the data in the projdata object 
    passed in the constructor. 
  */
  virtual void apply(RelatedViewgrams<float>& viewgrams) const;

  //! Undo the normalisation of some data
  /*! 
    This means \c divide with the data in the projdata object 
    passed in the constructor. 
  */
  virtual void undo(RelatedViewgrams<float>& viewgrams) const;

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
  shared_ptr<ProjDataInfo> proj_data_info_ptr;
  ProjDataInfoCylindricalNoArcCorr const * proj_data_info_cyl_ptr;
  shared_ptr<ProjDataInfoCylindricalNoArcCorr> proj_data_info_cyl_uncompressed_ptr;
  int span;
  int mash;

  void read_norm_data(const string& filename);
  float get_deadtime_efficiency ( const DetectionPosition<>& det_pos) const;

  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  string normalisation_ECAT7_filename;
};

END_NAMESPACE_ECAT7  
END_NAMESPACE_STIR

#endif
