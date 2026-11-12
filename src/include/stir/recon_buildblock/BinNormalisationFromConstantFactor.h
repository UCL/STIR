//
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisationFromConstantfactor

  \author Viet Dao
*/
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_BinNormalisationFromConstantFactor_H__
#define __stir_recon_buildblock_BinNormalisationFromConstantFactor_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RelatedViewgrams.h"
START_NAMESPACE_STIR

/*!
  \ingroup normalisation
  \brief A BinNormalisation class that gets the normalisation factors from
  a constant factor such as time scale or decay factors.
*/

class BinNormalisationFromConstantFactor : public BinNormalisation
{
public:
  explicit BinNormalisationFromConstantFactor(const float factor)
      : _factor(factor)
  {}

  bool is_trivial() const override { return this->_factor == 1.F; }

  std::string get_registered_name() const override { return "Constant factor"; }

  float get_bin_efficiency(const Bin&) const override { return this->_factor; }

  void apply(RelatedViewgrams<float>& viewgrams) const override { viewgrams /= this->_factor; }

  void undo(RelatedViewgrams<float>& viewgrams) const override { viewgrams *= this->_factor; }

private:
  float _factor;
};

END_NAMESPACE_STIR

#endif