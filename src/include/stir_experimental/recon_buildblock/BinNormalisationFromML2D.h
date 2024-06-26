//
//
/*
    Copyright (C) 2004- 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class stir::BinNormalisationFromML2D

  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_BinNormalisationFromML2D_H__
#define __stir_recon_buildblock_BinNormalisationFromML2D_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInMemory.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \ingroup recon_buildblock
  \brief A BinNormalisation class that gets the normalisation factors from
  the files output by find_ML_normfactors.

  \warning the ProjData object has to be 2D, no mashing, no span, no arc-correction.
  I'm not sure if this is properly checked at run-time.

  \par Parsing details

  Default values are given below, except for the filename.
\verbatim
  bin normalisation type := From ML2D
  Bin Normalisation From ML2D:=
  normalisation_filename_prefix:=<ASCII>
  use block factors:=1
  use geometric factors:=1
  use crystal_efficiencies:=1
  efficiency iteration number:=0
  iteration number:=0
  End Bin Normalisation From ML2D:=
\endverbatim


*/
class BinNormalisationFromML2D : public RegisteredParsingObject<BinNormalisationFromML2D, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char* const registered_name;

  //! Default constructor
  /*!
    \warning You should not call any member functions for any object just
    constructed with this constructor. Initialise the object properly first
    by parsing.
  */
  BinNormalisationFromML2D();

  //! Checks if we can handle certain projection data.
  virtual Succeeded set_up(const shared_ptr<const ProjDataInfo>&);

  //! Normalise some data
  /*!
    This means \c multiply with the data in the projdata object
    passed in the constructor.
  */
  void apply(RelatedViewgrams<float>& viewgrams) const override;

  //! Undo the normalisation of some data
  /*!
    This means \c divide with the data in the projdata object
    passed in the constructor.
  */
  void undo(RelatedViewgrams<float>& viewgrams) const override;

private:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

  std::string normalisation_filename_prefix;
  bool do_block;
  bool do_geo;
  bool do_eff;
  int eff_iter_num;
  int iter_num;
  // use shared pointer to avoid calling default constructor
  shared_ptr<ProjDataInMemory> norm_factors_ptr;
};

END_NAMESPACE_STIR

#endif
