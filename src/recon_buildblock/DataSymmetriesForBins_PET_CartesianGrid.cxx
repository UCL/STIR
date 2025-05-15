//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2018, Palak Wadhwa
    Copyright (C) 2021-2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup symmetries
  \brief non-inline implementations for class stir::DataSymmetriesForBins_PET_CartesianGrid

  \author Kris Thielemans
  \author Palak Wadhwa
  \author PARAPET project
  \author Parisa Khateri


*/
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/ProjDataInfoSubsetByView.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/shared_ptr.h"
#include "stir/round.h"
#include <typeinfo>
#include <algorithm>
#include "stir/format.h"
#include "stir/ProjDataInfoBlocksOnCylindrical.h"
#include "stir/ProjDataInfoGeneric.h"
#include "stir/warning.h"
#include "stir/error.h"

using std::min;
using std::max;

START_NAMESPACE_STIR

//! find correspondence between axial_pos_num and image coordinates
/*! z = num_planes_per_axial_pos * axial_pos_num + axial_pos_to_z_offset
   compute the offset by matching up the centre of the scanner
   in the 2 coordinate systems
*/
static void
find_relation_between_coordinate_systems(int& num_planes_per_scanner_ring,
                                         VectorWithOffset<int>& num_planes_per_axial_pos,
                                         VectorWithOffset<float>& axial_pos_to_z_offset,
                                         const ProjDataInfoCylindrical* proj_data_info_cyl_ptr,
                                         const DiscretisedDensityOnCartesianGrid<3, float>* cartesian_grid_info_ptr)

{

  const int min_segment_num = proj_data_info_cyl_ptr->get_min_segment_num();
  const int max_segment_num = proj_data_info_cyl_ptr->get_max_segment_num();

  num_planes_per_axial_pos = VectorWithOffset<int>(min_segment_num, max_segment_num);
  axial_pos_to_z_offset = VectorWithOffset<float>(min_segment_num, max_segment_num);

  // TODO and WARNING: get_grid_spacing()[1] is z()
  const float image_plane_spacing = cartesian_grid_info_ptr->get_grid_spacing()[1];

  {
    const float num_planes_per_scanner_ring_float = proj_data_info_cyl_ptr->get_ring_spacing() / image_plane_spacing;

    num_planes_per_scanner_ring = round(num_planes_per_scanner_ring_float);

    if (fabs(num_planes_per_scanner_ring_float - num_planes_per_scanner_ring) > 1.E-2)
      error(format("DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing "
                   "equal to the ring spacing of the scanner divided by an integer. Sorry. "
                   "(Image z-spacing is {} and ring spacing is {})",
                   image_plane_spacing,
                   proj_data_info_cyl_ptr->get_ring_spacing()));
  }

  /* disabled as we support this now
  if (fabs( cartesian_grid_info_ptr->get_origin().x()) > 1.E-2)
    error("DataSymmetriesForBins_PET_CartesianGrid can currently only support x-origin = 0 "
          "Sorry\n");
  if (fabs( cartesian_grid_info_ptr->get_origin().y()) > 1.E-2)
    error("DataSymmetriesForBins_PET_CartesianGrid can currently only support y-origin = 0 "
          "Sorry\n");
  */

  for (int segment_num = min_segment_num; segment_num <= max_segment_num; ++segment_num)
    {
      {
        const float num_planes_per_axial_pos_float
            = proj_data_info_cyl_ptr->get_axial_sampling(segment_num) / image_plane_spacing;

        num_planes_per_axial_pos[segment_num] = round(num_planes_per_axial_pos_float);

        if (fabs(num_planes_per_axial_pos_float - num_planes_per_axial_pos[segment_num]) > 1.E-2)
          error(format("DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing "
                       "equal to the sinogram spacing of the scanner divided by an integer. Sorry. "
                       "(Image z-spacing is {} and axial sinogram spacing is {} at segment {}",
                       image_plane_spacing,
                       proj_data_info_cyl_ptr->get_axial_sampling(segment_num),
                       segment_num));
      }

      const float delta = proj_data_info_cyl_ptr->get_average_ring_difference(segment_num);

      // KT 20/06/2001 take origin.z() into account
      axial_pos_to_z_offset[segment_num]
          = (cartesian_grid_info_ptr->get_max_index() + cartesian_grid_info_ptr->get_min_index()) / 2.F
            - cartesian_grid_info_ptr->get_origin().z() / image_plane_spacing
            - (num_planes_per_axial_pos[segment_num]
                   * (proj_data_info_cyl_ptr->get_max_axial_pos_num(segment_num)
                      + proj_data_info_cyl_ptr->get_min_axial_pos_num(segment_num))
               + num_planes_per_scanner_ring * delta)
                  / 2;
    }
}

#if 0 // disabled as currently the same as ProjDataInfoGeneric
//overload for block geometry
static void
find_relation_between_coordinate_systems(int& num_planes_per_scanner_ring,
                                         VectorWithOffset<int>& num_planes_per_axial_pos,
                                         VectorWithOffset<float>& axial_pos_to_z_offset,
                                         const ProjDataInfoBlocksOnCylindrical* proj_data_info_blk_ptr,
                                         const DiscretisedDensityOnCartesianGrid<3,float> *  cartesian_grid_info_ptr)

{
  const int min_segment_num = proj_data_info_blk_ptr->get_min_segment_num();
  const int max_segment_num = proj_data_info_blk_ptr->get_max_segment_num();

  num_planes_per_axial_pos = VectorWithOffset<int>(min_segment_num, max_segment_num);
  axial_pos_to_z_offset = VectorWithOffset<float>(min_segment_num, max_segment_num);

  // TODO and WARNING: get_grid_spacing()[1] is z()
  const float image_plane_spacing = cartesian_grid_info_ptr->get_grid_spacing()[1];

  {
    const float num_planes_per_scanner_ring_float =
      proj_data_info_blk_ptr->get_ring_spacing() / image_plane_spacing;

    num_planes_per_scanner_ring = round(num_planes_per_scanner_ring_float);
    //parisa: temporarily comment
    /*if (fabs(num_planes_per_scanner_ring_float - num_planes_per_scanner_ring) > 1.E-2)
      error("DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing "
           "equal to the ring spacing of the scanner divided by an integer. Sorry\n");*/
  }

  /* disabled as we support this now
  if (fabs( cartesian_grid_info_ptr->get_origin().x()) > 1.E-2)
    error("DataSymmetriesForBins_PET_CartesianGrid can currently only support x-origin = 0 "
         "Sorry\n");
  if (fabs( cartesian_grid_info_ptr->get_origin().y()) > 1.E-2)
    error("DataSymmetriesForBins_PET_CartesianGrid can currently only support y-origin = 0 "
         "Sorry\n");
  */

  for (int segment_num=min_segment_num; segment_num<=max_segment_num; ++segment_num)
  {
    {
      const float
        num_planes_per_axial_pos_float =
        proj_data_info_blk_ptr->get_axial_sampling(segment_num)/image_plane_spacing;

      num_planes_per_axial_pos[segment_num] = round(num_planes_per_axial_pos_float);
      //parisa: temporarily comment
      /*if (fabs(num_planes_per_axial_pos_float - num_planes_per_axial_pos[segment_num]) > 1.E-5)
        error("DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing "
             "equal to the axial sampling in the projection data divided by an integer. Sorry\n");*/

    }

    const float delta = proj_data_info_blk_ptr->get_average_ring_difference(segment_num);

    // KT 20/06/2001 take origin.z() into account
    axial_pos_to_z_offset[segment_num] =
      (cartesian_grid_info_ptr->get_max_index() + cartesian_grid_info_ptr->get_min_index())/2.F
      - cartesian_grid_info_ptr->get_origin().z()/image_plane_spacing
      -
      (num_planes_per_axial_pos[segment_num]
       *(proj_data_info_blk_ptr->get_max_axial_pos_num(segment_num)
         + proj_data_info_blk_ptr->get_min_axial_pos_num(segment_num))
       + num_planes_per_scanner_ring*delta)/2;
  }
}
#endif

#if 0 // disabled as now derived from Cylindrical
// overloading for generic case
static void
find_relation_between_coordinate_systems(int& num_planes_per_scanner_ring,
                                         VectorWithOffset<int>& num_planes_per_axial_pos,
                                         VectorWithOffset<float>& axial_pos_to_z_offset,
                                         const ProjDataInfoGeneric* proj_data_info_blk_ptr,
                                         const DiscretisedDensityOnCartesianGrid<3,float> *  cartesian_grid_info_ptr)
{
    const int min_segment_num = proj_data_info_blk_ptr->get_min_segment_num();
    const int max_segment_num = proj_data_info_blk_ptr->get_max_segment_num();

    num_planes_per_axial_pos = VectorWithOffset<int>(min_segment_num, max_segment_num);
    axial_pos_to_z_offset = VectorWithOffset<float>(min_segment_num, max_segment_num);

    // TODO and WARNING: get_grid_spacing()[1] is z()
    const float image_plane_spacing = cartesian_grid_info_ptr->get_grid_spacing()[1];

  
    const float num_planes_per_scanner_ring_float =
            proj_data_info_blk_ptr->get_ring_spacing() / image_plane_spacing;

    num_planes_per_scanner_ring = round(num_planes_per_scanner_ring_float);


    for (int segment_num=min_segment_num; segment_num<=max_segment_num; ++segment_num)
    {
        const float num_planes_per_axial_pos_float =
                proj_data_info_blk_ptr->get_axial_sampling(segment_num)/image_plane_spacing;
                
        num_planes_per_axial_pos[segment_num] = round(num_planes_per_axial_pos_float);

        const float delta = proj_data_info_blk_ptr->get_average_ring_difference(segment_num);
        axial_pos_to_z_offset[segment_num] =
                (cartesian_grid_info_ptr->get_max_index() + cartesian_grid_info_ptr->get_min_index())/2.F
                - cartesian_grid_info_ptr->get_origin().z()/image_plane_spacing
                - (num_planes_per_axial_pos[segment_num]
                *(proj_data_info_blk_ptr->get_max_axial_pos_num(segment_num)
                + proj_data_info_blk_ptr->get_min_axial_pos_num(segment_num))
                + num_planes_per_scanner_ring*delta)/2;
    }
}
#endif

/*! The DiscretisedDensity pointer has to point to an object of
  type  DiscretisedDensityOnCartesianGrid (or a derived type).

  We really need only the geometrical info from the image. At the moment
  we have to use the data itself as well.
*/
DataSymmetriesForBins_PET_CartesianGrid::DataSymmetriesForBins_PET_CartesianGrid(
    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<const DiscretisedDensity<3, float>>& image_info_ptr,
    const bool do_symmetry_90degrees_min_phi_v,
    const bool do_symmetry_180degrees_min_phi_v,
    const bool do_symmetry_swap_segment_v,
    const bool do_symmetry_swap_s_v,
    const bool do_symmetry_shift_z)
    : DataSymmetriesForBins(proj_data_info_ptr),
      do_symmetry_90degrees_min_phi(do_symmetry_90degrees_min_phi_v),
      do_symmetry_180degrees_min_phi(do_symmetry_90degrees_min_phi_v || do_symmetry_180degrees_min_phi_v),
      do_symmetry_swap_segment(do_symmetry_swap_segment_v),
      do_symmetry_swap_s(do_symmetry_swap_s_v),
      do_symmetry_shift_z(do_symmetry_shift_z)
{
  auto subset_proj_data_info_ptr = dynamic_cast<const ProjDataInfoSubsetByView*>(proj_data_info_ptr.get());
  if (!is_null_ptr(subset_proj_data_info_ptr))
    {
      // special handling of subset case
      // will for now just switch view syms off
      if (is_null_ptr(
              dynamic_cast<const ProjDataInfoCylindrical*>(subset_proj_data_info_ptr->get_original_proj_data_info_sptr().get())))
        error("DataSymmetriesForBins_PET_CartesianGrid constructed with wrong type of original (non-subset) ProjDataInfo: %s\n"
              "(can only handle projection data corresponding to a cylinder)\n",
              typeid(*subset_proj_data_info_ptr->get_original_proj_data_info_sptr()).name());

      if (do_symmetry_90degrees_min_phi || do_symmetry_180degrees_min_phi)
        {
          warning("Turning off 90 and 180 degrees minus phi symmetries for subsets.");
        }
      do_symmetry_90degrees_min_phi = false;
      do_symmetry_180degrees_min_phi = false;
    }

  auto pdi_cyl_ptr = dynamic_cast<const ProjDataInfoCylindrical*>(
      subset_proj_data_info_ptr ? subset_proj_data_info_ptr->get_original_proj_data_info_sptr().get() : proj_data_info_ptr.get());
  initialise_deltas(pdi_cyl_ptr);

  if (proj_data_info_ptr->get_scanner_ptr()->get_scanner_geometry() == "Cylindrical")
    {
      if (dynamic_cast<const ProjDataInfoCylindrical*>(pdi_cyl_ptr) == NULL)
        error("DataSymmetriesForBins_PET_CartesianGrid constructed with wrong type of ProjDataInfo: %s\n"
              "(can only handle projection data corresponding to a cylinder)\n",
              typeid(*pdi_cyl_ptr).name());

      const DiscretisedDensityOnCartesianGrid<3, float>* cartesian_grid_info_ptr
          = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, float>*>(image_info_ptr.get());

      if (is_null_ptr(cartesian_grid_info_ptr))
        error("DataSymmetriesForBins_PET_CartesianGrid constructed with wrong type of image info: %s\n",
              typeid(*image_info_ptr).name());

      // WARNING get_grid_spacing()[1] == z
      const float z_origin_in_planes = image_info_ptr->get_origin().z() / cartesian_grid_info_ptr->get_grid_spacing()[1];
      // z_origin_in_planes should be an integer
      if (fabs(round(z_origin_in_planes) - z_origin_in_planes) > 1.E-3F)
        error("DataSymmetriesForBins_PET_CartesianGrid: the shift in the "
              "z-direction of the origin (which is %g) should be a multiple of the plane "
              "separation (%g)\n",
              image_info_ptr->get_origin().z(),
              cartesian_grid_info_ptr->get_grid_spacing()[1]);

      // check if unequal voxel size in x,y, if so, use less symmetry
      if (fabs(cartesian_grid_info_ptr->get_grid_spacing()[2] - cartesian_grid_info_ptr->get_grid_spacing()[3]) > 2.E-3F)
        do_symmetry_90degrees_min_phi = false;

      num_views = proj_data_info_ptr->get_num_views();

      if (num_views % 4 != 0)
        do_symmetry_90degrees_min_phi = false;

      if (num_views % 2 != 0)
        do_symmetry_180degrees_min_phi = false;

      // check on segment symmetry
      if (fabs(proj_data_info_ptr->get_tantheta(Bin(0, 0, 0, 0))) > 1.E-4F)
        error("DataSymmetriesForBins_PET_CartesianGrid can only handle projection data "
              "with segment 0 corresponding to direct planes (i.e. theta==0)\n");

      for (int segment_num = 1;
           segment_num <= min(proj_data_info_ptr->get_max_segment_num(), -proj_data_info_ptr->get_min_segment_num());
           ++segment_num)
        if (fabs(proj_data_info_ptr->get_tantheta(Bin(segment_num, 0, 0, 0))
                 + proj_data_info_ptr->get_tantheta(Bin(-segment_num, 0, 0, 0)))
            > 1.E-4F)
          error("DataSymmetriesForBins_PET_CartesianGrid can only handle projection data "
                "with negative segment numbers corresponding to -theta of the positive segments. "
                "This is not true for segment pair %d.\n",
                segment_num);

      // feable check on s-symmetry
      if (fabs(proj_data_info_ptr->get_s(Bin(0, 0, 0, 1)) + proj_data_info_ptr->get_s(Bin(0, 0, 0, -1))) > 1.E-4F)
        error("DataSymmetriesForBins_PET_CartesianGrid can only handle projection data "
              "with tangential_pos_num s.t. get_s(...,tang_pos_num)==-get_s(...,-tang_pos_num)\n");

      // PW Disabling some symmetries due to phi offset.
      if (fabs(proj_data_info_ptr->get_phi(Bin(0, 0, 0, 0))) > 1.E-4F
          && (this->do_symmetry_90degrees_min_phi || this->do_symmetry_180degrees_min_phi))
        {
          info("Disabling symmetries for the projector as image is rotated due to phi offset of the scanner.");
          this->do_symmetry_90degrees_min_phi = false;
          this->do_symmetry_180degrees_min_phi = false;
        }

      // RT Disabling some symmetries due to tof data
      if (proj_data_info_ptr->is_tof_data())
        {
          if (this->do_symmetry_90degrees_min_phi || this->do_symmetry_180degrees_min_phi)
            {
              info("Disabling rotational symmetries for the projector with TOF data as this is untested.");
              this->do_symmetry_90degrees_min_phi = false;
              this->do_symmetry_180degrees_min_phi = false;
            }

          if (this->do_symmetry_swap_segment)
            {
              info("Disabling segment swapping for the projector with TOF data as this is untested.");
              this->do_symmetry_swap_segment = false;
            }

          if (this->do_symmetry_swap_s)
            {
              info("Disabling swap s symmetry for the projector with TOF data as this is untested.");
              this->do_symmetry_swap_s = false;
            }
        }

      if (fabs(image_info_ptr->get_origin().x()) > .01F || fabs(image_info_ptr->get_origin().y()) > .01F)
        {
          // disable symmetries with shifted images
          if (this->do_symmetry_90degrees_min_phi || this->do_symmetry_180degrees_min_phi || this->do_symmetry_swap_segment
              || this->do_symmetry_swap_s)
            {
              info("Disabling symmetries for the projector in transaxial plane as image is shifted");
              this->do_symmetry_90degrees_min_phi = this->do_symmetry_180degrees_min_phi = this->do_symmetry_swap_segment
                  = this->do_symmetry_swap_s = false;
            }
        }
      find_relation_between_coordinate_systems(
          num_planes_per_scanner_ring, num_planes_per_axial_pos, axial_pos_to_z_offset, pdi_cyl_ptr, cartesian_grid_info_ptr);
    }
  // Block implementation
  if (proj_data_info_ptr->get_scanner_ptr()->get_scanner_geometry() == "BlocksOnCylindrical")
    {
      if (dynamic_cast<const ProjDataInfoBlocksOnCylindrical*>(pdi_cyl_ptr) == NULL)
        error("DataSymmetriesForBins_PET_CartesianGrid constructed with wrong type of ProjDataInfo: %s\n"
              "(can only handle projection data corresponding to blocks on a cylinder)\n",
              typeid(*pdi_cyl_ptr).name());

      const DiscretisedDensityOnCartesianGrid<3, float>* cartesian_grid_info_ptr
          = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, float>*>(image_info_ptr.get());

      if (cartesian_grid_info_ptr == NULL)
        error("DataSymmetriesForBins_PET_CartesianGrid constructed with wrong type of image info: %s\n",
              typeid(*image_info_ptr).name());

      // WARNING get_grid_spacing()[1] == z
      // note: origin by default is (0,0,0)
      const float z_origin_in_planes = image_info_ptr->get_origin().z() / cartesian_grid_info_ptr->get_grid_spacing()[1];
      // z_origin_in_planes should be an integer
      if (fabs(round(z_origin_in_planes) - z_origin_in_planes) > 1.E-3F)
        error("DataSymmetriesForBins_PET_CartesianGrid: the shift in the "
              "z-direction of the origin (which is %g) should be a multiple of the plane "
              "separation (%g)\n",
              image_info_ptr->get_origin().z(),
              cartesian_grid_info_ptr->get_grid_spacing()[1]);

      if (this->do_symmetry_90degrees_min_phi || this->do_symmetry_180degrees_min_phi || this->do_symmetry_swap_segment
          || this->do_symmetry_swap_s)
        {
          info("Disabling all symmetries for the projector except for symmetry_z since they are not implemented in block "
               "geometry yet.");
          this->do_symmetry_90degrees_min_phi = this->do_symmetry_180degrees_min_phi = this->do_symmetry_swap_segment
              = this->do_symmetry_swap_s = this->do_symmetry_shift_z = false;
        }
      if (!dynamic_cast<const ProjDataInfoBlocksOnCylindrical*>(pdi_cyl_ptr)->axial_sampling_is_uniform())
        {
          this->do_symmetry_shift_z = false;
          this->do_symmetry_swap_segment = false;
        }

      // TODOBLOCK. should probably not call next function for non-uniform sampling
      find_relation_between_coordinate_systems(
          num_planes_per_scanner_ring, num_planes_per_axial_pos, axial_pos_to_z_offset, pdi_cyl_ptr, cartesian_grid_info_ptr);
    }
  // generic implementation
  if (proj_data_info_ptr->get_scanner_ptr()->get_scanner_geometry() == "Generic")
    {
      if (dynamic_cast<const ProjDataInfoGeneric*>(pdi_cyl_ptr) == NULL)
        error("DataSymmetriesForBins_PET_CartesianGrid constructed with wrong type of ProjDataInfo: %s\n"
              "(can only handle projection data corresponding to a generig geometry)\n",
              typeid(*pdi_cyl_ptr).name());

      const DiscretisedDensityOnCartesianGrid<3, float>* cartesian_grid_info_ptr
          = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, float>*>(image_info_ptr.get());

      if (cartesian_grid_info_ptr == NULL)
        error("DataSymmetriesForBins_PET_CartesianGrid constructed with wrong type of image info: %s\n",
              typeid(*image_info_ptr).name());

      // WARNING get_grid_spacing()[1] == z
      const float z_origin_in_planes = image_info_ptr->get_origin().z() / cartesian_grid_info_ptr->get_grid_spacing()[1];
      // z_origin_in_planes should be an integer
      if (fabs(round(z_origin_in_planes) - z_origin_in_planes) > 1.E-3F)
        error("DataSymmetriesForBins_PET_CartesianGrid: the shift in the "
              "z-direction of the origin (which is %g) should be a multiple of the plane "
              "separation (%g)\n",
              image_info_ptr->get_origin().z(),
              cartesian_grid_info_ptr->get_grid_spacing()[1]);

      if (this->do_symmetry_90degrees_min_phi || this->do_symmetry_180degrees_min_phi || this->do_symmetry_swap_segment
          || this->do_symmetry_swap_s || this->do_symmetry_shift_z)
        {
          info("Disabling all symmetries for the projector since they are not implemented in generic geometry.");
          this->do_symmetry_90degrees_min_phi = this->do_symmetry_180degrees_min_phi = this->do_symmetry_swap_segment
              = this->do_symmetry_swap_s = this->do_symmetry_shift_z = false;
        }

      if (!dynamic_cast<const ProjDataInfoGeneric*>(pdi_cyl_ptr)->axial_sampling_is_uniform())
        {
          this->do_symmetry_shift_z = false;
          this->do_symmetry_swap_segment = false;
        }

      // TODOBLOCK. should probably not call next function for non-uniform sampling
      find_relation_between_coordinate_systems(
          num_planes_per_scanner_ring, num_planes_per_axial_pos, axial_pos_to_z_offset, pdi_cyl_ptr, cartesian_grid_info_ptr);
    }
}

void
DataSymmetriesForBins_PET_CartesianGrid::initialise_deltas(const ProjDataInfoCylindrical* pdi_ptr)
{
  this->deltas.resize(pdi_ptr->get_min_segment_num(), pdi_ptr->get_max_segment_num());
  for (int segment_num = pdi_ptr->get_min_segment_num(); segment_num <= pdi_ptr->get_max_segment_num(); ++segment_num)
    {
      this->deltas[segment_num] = pdi_ptr->get_average_ring_difference(segment_num);
    }
}

#ifndef STIR_NO_COVARIANT_RETURN_TYPES
DataSymmetriesForBins_PET_CartesianGrid*
#else
DataSymmetriesForViewSegmentNumbers*
#endif
DataSymmetriesForBins_PET_CartesianGrid::clone() const
{
  return new DataSymmetriesForBins_PET_CartesianGrid(*this);
}

bool
DataSymmetriesForBins_PET_CartesianGrid::operator==(const DataSymmetriesForBins_PET_CartesianGrid& sym) const
{
  if (!base_type::operator==(sym))
    return false;

  return this->do_symmetry_90degrees_min_phi == sym.do_symmetry_90degrees_min_phi
         && this->do_symmetry_180degrees_min_phi == sym.do_symmetry_180degrees_min_phi
         && this->do_symmetry_swap_segment == sym.do_symmetry_swap_segment && this->do_symmetry_swap_s == sym.do_symmetry_swap_s
         && this->do_symmetry_shift_z == sym.do_symmetry_shift_z && this->num_views == sym.num_views
         && this->num_planes_per_scanner_ring == sym.num_planes_per_scanner_ring
         && this->num_planes_per_axial_pos == sym.num_planes_per_axial_pos
         && this->axial_pos_to_z_offset == sym.axial_pos_to_z_offset;
}

bool
DataSymmetriesForBins_PET_CartesianGrid::blindly_equals(const root_type* const that_ptr) const
{
  assert(dynamic_cast<const self_type* const>(that_ptr) != 0);
  return this->operator==(static_cast<const self_type&>(*that_ptr));
}

END_NAMESPACE_STIR
