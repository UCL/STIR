//
// $Id: BackProjectorByBinUsingSquareProjMatrixByBin.h
//

#ifndef _BackProjectorByBinUsingSquareProjMatrixByBin_
#define _BackProjectorByBinUsingSquareProjMatrixByBin_

/*!

  \file

  \brief

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2001, IRSL
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
//#include "stir/recon_buildblock/DataSymmetriesForBins.h"
//#include "stir/RelatedViewgrams.h"

class Viewgrams;
template <typename elemT>
class RelatedViewgrams;
class ProjDataInfoCylindricalArcCorr;

START_NAMESPACE_STIR

/*!
  \brief This implements the BackProjectorByBin interface, given any
ProjMatrixByBin object

  */
class BackProjectorByBinUsingSquareProjMatrixByBin
    : public RegisteredParsingObject<BackProjectorByBinUsingSquareProjMatrixByBin, BackProjectorByBin>
{
public:
  static const char* const registered_name;

  BackProjectorByBinUsingSquareProjMatrixByBin();

  BackProjectorByBinUsingSquareProjMatrixByBin(const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr);

  const DataSymmetriesForViewSegmentNumbers* get_symmetries_used() const override;

  void set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
              const shared_ptr<const DiscretisedDensity<3, float>>& density_info_ptr // TODO should be Info only
              ) override;

  void actual_back_project(DiscretisedDensity<3, float>& image,
                           const RelatedViewgrams<float>&,
                           const int min_axial_pos_num,
                           const int max_axial_pos_num,
                           const int min_tangential_pos_num,
                           const int max_tangential_pos_num) override;

  shared_ptr<ProjMatrixByBin>& get_proj_matrix_sptr() { return proj_matrix_ptr; }

protected:
  shared_ptr<ProjMatrixByBin> proj_matrix_ptr;

  void actual_back_project(DiscretisedDensity<3, float>& image, const Bin& bin);

private:
  void set_defaults() override;
  void initialise_keymap() override;
};

END_NAMESPACE_STIR

//#include "stir/BackProjectorByBinUsingSquareProjMatrixByBin.inl"

#endif
