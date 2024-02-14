//
//
/*!
  \file
  \ingroup recon_buildblock

  \brief ProjMatrixByBinSinglePhoton's definition

  \author Kris

*/
/*
    Copyright (C) 2000- 2003, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_ProjMatrixByBinSinglePhoton__
#define __stir_recon_buildblock_ProjMatrixByBinSinglePhoton__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief a 'projection matrix' to implement a model for a single
  photon acquisition in terms of the detector efficiencies.

  \todo This is a horrible work-around for the fact that STIR currently
  insists on working on a density.

*/

class ProjMatrixByBinSinglePhoton : public RegisteredParsingObject<ProjMatrixByBinSinglePhoton, ProjMatrixByBin, ProjMatrixByBin>
{
public:
  //! Name which will be used when parsing a ProjMatrixByBin object
  static const char* const registered_name;

  //! Default constructor (calls set_defaults())
  ProjMatrixByBinSinglePhoton();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
   */
  void set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
              const shared_ptr<const DiscretisedDensity<3, float>>& density_info_ptr // TODO should be Info only
              ) override;

private:
  // explicitly list necessary members for image details (should use an Info object instead)
  CartesianCoordinate3D<int> min_index;
  CartesianCoordinate3D<int> max_index;

  shared_ptr<const ProjDataInfo> proj_data_info_ptr;

  void calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin&) const override;

  void set_defaults() override;
  void initialise_keymap() override;
};

END_NAMESPACE_STIR

#endif
