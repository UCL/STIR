/*
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup simset
  \brief Rebins a SimSET block-detector weight file, binned by crystal pair, into STIR projection data

  SimSET run with a block detector and <tt>bin_by_crystal = true</tt> writes its
  weights indexed by <i>crystal pair</i>, not by sinogram bin. This utility maps each
  bin of a STIR template onto its detection position pairs, and sums the
  corresponding crystal-pair weights.

  \par Usage
  \code
  conv_SimSET_crystal_pairs_to_STIR output_prefix template_projdata weight_file mode \
      num_block_rings num_blocks_per_ring crystals_per_block_axial crystals_per_block_transaxial
  \endcode
  \param output_prefix outputs are written as <tt>output_prefix_<label>.hs</tt>
  \param template_projdata projection data describing the scanner and the output sinogram
  \param weight_file SimSET's <tt>weight_image_path</tt> from the \c bin.rec file
  \param mode \c multiples or \c singles_scat, see below

  The last four arguments describe the simulated SimSET detector, and have to agree with
  the template's scanner:
  - \c num_block_rings : number of block rings in \c det.rec that contain active crystals
    (rings with only shielding do not count)
  - \c num_blocks_per_ring : \c ring_num_blocks_in_ring in the ring parameter file
  - \c crystals_per_block_axial : number of active crystals per block along z
  - \c crystals_per_block_transaxial : number of active crystals per block along y

  When every element of the block is an active crystal, the last two are
  <tt>block_layer_num_z_changes + 1</tt> and <tt>block_layer_num_y_changes + 1</tt>.

  \par File format and modes
  A weight file is a SimSET header of 32768 bytes, followed by one or more grids
  of <tt>num_crystals x num_crystals</tt> native-endian \c floats. How many grids
  there are was decided at simulation time by the \c bin.rec file.
  - \c multiples expects 1 grid, written as label \c multiples.
  - \c singles_scat expects 2 grids, <tt>[unscattered][single scatter]</tt>
    (e.g. <tt>scatter_param = 5</tt>, <tt>min_s = max_s = 2</tt>), written as labels
    \c unscattered and \c singles_scat.
  The number of grids in the file has to match the mode.

  \par Crystal numbering
  The template's scanner has to describe the simulated detector, <i>including its
  block subdivision</i>. The crystal index of a detection position is
  \code
  block_index * (crystals_per_block_axial * crystals_per_block_transaxial)
     + axial_index_in_block * crystals_per_block_transaxial + transaxial_index_in_block
  \endcode
  with <tt>block_index = axial_block * num_transaxial_blocks + transaxial_block</tt>.
  A template with the right total number of crystals but a different block
  subdivision would give a permuted, wrong, sinogram without any error, which is
  why the SimSET detector has to be given on the command line.
  Only the upper triangle of each grid (<tt>crystal1 < crystal2</tt>) is used.

  The views of the output are stored in reverse order (<tt>view -> num_views - 1 - view</tt>).
  SimSET's y axis points up (voxel row 0 is at \c yMax) and STIR's points down, so an image
  written row by row for SimSET is the same picture in both, with <tt>y_SimSET = -y_STIR</tt>.
  Both number their crystals counter-clockwise in their own frame, so seen from the object the
  two rings are numbered in opposite directions: SimSET crystal \c k is the mirror image of STIR
  detector \c k. Reversing the views undoes this mirror. The exact alignment depends on the
  template's view offset.

  \warning The whole grid is read into memory (4 num_crystals^2 bytes, 5.8 GB for 38016 crystals).
  \warning Only Cylindrical and BlocksOnCylindrical non-arc-corrected, non-TOF, templates are supported.
*/

#include "stir/ProjData.h"
#include "stir/ProjDataInMemory.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/DetectionPositionPair.h"
#include "stir/Bin.h"
#include "stir/Scanner.h"
#include "stir/format.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace stir;

namespace
{
//! size of the header that SimSET writes before the binned data
constexpr std::uint64_t simset_header_bytes = 32768;

//! maps a STIR detection position onto SimSET's block-major crystal index
struct CrystalIndexer
{
  int crystals_per_block_axial;
  int crystals_per_block_trans;
  int num_blocks_trans;

  int operator()(const DetectionPosition<>& pos) const
  {
    const int ring = pos.axial_coord();
    const int det = pos.tangential_coord();
    const int block_index = (ring / crystals_per_block_axial) * num_blocks_trans + det / crystals_per_block_trans;
    const int index_in_block = (ring % crystals_per_block_axial) * crystals_per_block_trans + det % crystals_per_block_trans;
    return block_index * (crystals_per_block_axial * crystals_per_block_trans) + index_in_block;
  }
};

std::vector<float>
read_grid(std::ifstream& in, const int grid_num, const std::uint64_t num_crystals)
{
  const std::uint64_t num_floats = num_crystals * num_crystals;
  std::vector<float> grid(num_floats);
  in.clear();
  in.seekg(static_cast<std::streamoff>(simset_header_bytes + grid_num * num_floats * sizeof(float)));
  in.read(reinterpret_cast<char*>(grid.data()), static_cast<std::streamsize>(num_floats * sizeof(float)));
  if (!in)
    error(format("conv_SimSET_crystal_pairs_to_STIR: error reading grid {} from the weight file", grid_num));
  return grid;
}

void
rebin_grid(ProjDataInMemory& proj_data,
           const ProjDataInfoCylindricalNoArcCorr& proj_data_info,
           const std::vector<float>& grid,
           const std::uint64_t num_crystals,
           const CrystalIndexer& crystal_index)
{
  const int num_views = proj_data_info.get_num_views();

  // get_all_det_pos_pairs_for_bin() initialises a look-up table on first use.
  // Do that here, before the parallel loop.
  {
    std::vector<DetectionPositionPair<>> det_pos_pairs;
    const int seg = proj_data_info.get_min_segment_num();
    proj_data_info.get_all_det_pos_pairs_for_bin(
        det_pos_pairs, Bin(seg, 0, proj_data_info.get_min_axial_pos_num(seg), proj_data_info.get_min_tangential_pos_num()));
  }

#ifdef STIR_OPENMP
#  pragma omp parallel for collapse(2) schedule(dynamic)
#endif
  for (int seg = proj_data_info.get_min_segment_num(); seg <= proj_data_info.get_max_segment_num(); ++seg)
    for (int view = 0; view < num_views; ++view)
      {
        std::vector<DetectionPositionPair<>> det_pos_pairs;
        for (int ax = proj_data_info.get_min_axial_pos_num(seg); ax <= proj_data_info.get_max_axial_pos_num(seg); ++ax)
          for (int tang = proj_data_info.get_min_tangential_pos_num(); tang <= proj_data_info.get_max_tangential_pos_num();
               ++tang)
            {
              proj_data_info.get_all_det_pos_pairs_for_bin(det_pos_pairs, Bin(seg, view, ax, tang));
              float sum = 0.F;
              for (const auto& det_pos_pair : det_pos_pairs)
                {
                  std::uint64_t c1 = crystal_index(det_pos_pair.pos1());
                  std::uint64_t c2 = crystal_index(det_pos_pair.pos2());
                  if (c2 < c1)
                    std::swap(c1, c2);
                  sum += grid[c1 * num_crystals + c2];
                }
              // every output bin is written by exactly one iteration, so no synchronisation is needed
              proj_data.set_bin_value(Bin(seg, num_views - 1 - view, ax, tang, sum));
            }
      }
}

} // namespace

int
main(int argc, char** argv)
{
  if (argc != 9)
    {
      std::cerr << "Usage: " << argv[0] << " output_prefix template_projdata weight_file mode \\\n"
                << "    num_block_rings num_blocks_per_ring crystals_per_block_axial crystals_per_block_transaxial\n"
                << "  mode = multiples | singles_scat\n"
                << "    multiples    : 1-grid weight file, writes <output_prefix>_multiples.hs\n"
                << "    singles_scat : 2-grid weight file [unscattered][single scatter],\n"
                << "                   writes <output_prefix>_unscattered.hs and <output_prefix>_singles_scat.hs\n"
                << "  The last four describe the simulated SimSET detector:\n"
                << "    num_block_rings              : block rings in det.rec that contain active crystals\n"
                << "    num_blocks_per_ring          : ring_num_blocks_in_ring in the ring parameter file\n"
                << "    crystals_per_block_axial     : active crystals per block along z\n"
                << "    crystals_per_block_transaxial: active crystals per block along y\n";
      return EXIT_FAILURE;
    }

  const std::string output_prefix = argv[1];
  const std::string template_filename = argv[2];
  const std::string weight_filename = argv[3];
  const std::string mode = argv[4];
  const int simset_num_block_rings = std::stoi(argv[5]);
  const int simset_num_blocks_per_ring = std::stoi(argv[6]);
  const int simset_crystals_per_block_axial = std::stoi(argv[7]);
  const int simset_crystals_per_block_trans = std::stoi(argv[8]);

  std::vector<std::string> labels;
  if (mode == "multiples")
    labels = { "multiples" };
  else if (mode == "singles_scat")
    labels = { "unscattered", "singles_scat" };
  else
    error(format("conv_SimSET_crystal_pairs_to_STIR: unknown mode '{}'. Use 'multiples' or 'singles_scat'", mode));

  const auto template_sptr = ProjData::read_from_file(template_filename);
  const auto proj_data_info_sptr = template_sptr->get_proj_data_info_sptr();
  const auto proj_data_info_ptr = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get());
  if (!proj_data_info_ptr)
    error("conv_SimSET_crystal_pairs_to_STIR: the template has to be non-arc-corrected projection data");
  if (proj_data_info_ptr->is_tof_data())
    error("conv_SimSET_crystal_pairs_to_STIR: TOF templates are not supported");

  const Scanner& scanner = *proj_data_info_ptr->get_scanner_ptr();
  const auto geometry = scanner.get_scanner_geometry();
  if (geometry != "Cylindrical" && geometry != "BlocksOnCylindrical")
    error(format("conv_SimSET_crystal_pairs_to_STIR: unsupported scanner geometry '{}'", geometry));

  const int num_rings = scanner.get_num_rings();
  const int dets_per_ring = scanner.get_num_detectors_per_ring();
  const CrystalIndexer crystal_index{ scanner.get_num_axial_crystals_per_block(),
                                      scanner.get_num_transaxial_crystals_per_block(),
                                      dets_per_ring / scanner.get_num_transaxial_crystals_per_block() };
  if (simset_crystals_per_block_axial != crystal_index.crystals_per_block_axial
      || simset_crystals_per_block_trans != crystal_index.crystals_per_block_trans
      || simset_num_block_rings * simset_crystals_per_block_axial != num_rings
      || simset_num_blocks_per_ring * simset_crystals_per_block_trans != dets_per_ring)
    error(format("conv_SimSET_crystal_pairs_to_STIR: the template does not describe the SimSET detector.\n"
                 "SimSET: {} block rings of {} blocks of {} x {} (axial x transaxial) crystals\n"
                 "STIR template: {} rings of {} detectors, blocks of {} x {} (axial x transaxial) crystals",
                 simset_num_block_rings,
                 simset_num_blocks_per_ring,
                 simset_crystals_per_block_axial,
                 simset_crystals_per_block_trans,
                 num_rings,
                 dets_per_ring,
                 crystal_index.crystals_per_block_axial,
                 crystal_index.crystals_per_block_trans));

  const std::uint64_t num_crystals = static_cast<std::uint64_t>(num_rings) * dets_per_ring;

  info(format("conv_SimSET_crystal_pairs_to_STIR: {} rings x {} detectors per ring = {} crystals, "
              "blocks of {} x {} (axial x transaxial) crystals",
              num_rings,
              dets_per_ring,
              num_crystals,
              crystal_index.crystals_per_block_axial,
              crystal_index.crystals_per_block_trans));

  std::ifstream weight_file(weight_filename, std::ios::binary | std::ios::ate);
  if (!weight_file)
    error(format("conv_SimSET_crystal_pairs_to_STIR: cannot open weight file '{}'", weight_filename));
  const std::uint64_t file_size = static_cast<std::uint64_t>(weight_file.tellg());
  const std::uint64_t grid_bytes = num_crystals * num_crystals * sizeof(float);
  if (file_size < simset_header_bytes || (file_size - simset_header_bytes) % grid_bytes != 0)
    error(format("conv_SimSET_crystal_pairs_to_STIR: weight file size {} is not a {}-byte header plus a whole number of "
                 "{}-byte grids. Does the template describe the simulated detector?",
                 file_size,
                 simset_header_bytes,
                 grid_bytes));
  const std::uint64_t num_grids_in_file = (file_size - simset_header_bytes) / grid_bytes;
  if (num_grids_in_file != labels.size())
    error(format("conv_SimSET_crystal_pairs_to_STIR: weight file holds {} grid(s), but mode '{}' expects {}",
                 num_grids_in_file,
                 mode,
                 labels.size()));

  for (std::size_t grid_num = 0; grid_num < labels.size(); ++grid_num)
    {
      const auto grid = read_grid(weight_file, static_cast<int>(grid_num), num_crystals);

      // totals, to see how much of the grid ends up in the sinogram
      double upper_sum = 0., lower_sum = 0.;
#ifdef STIR_OPENMP
#  pragma omp parallel for reduction(+ : upper_sum, lower_sum) schedule(static)
#endif
      for (std::int64_t c1 = 0; c1 < static_cast<std::int64_t>(num_crystals); ++c1)
        for (std::uint64_t c2 = 0; c2 < num_crystals; ++c2)
          (c2 > static_cast<std::uint64_t>(c1) ? upper_sum : lower_sum) += grid[c1 * num_crystals + c2];
      if (upper_sum == 0.)
        warning(format("conv_SimSET_crystal_pairs_to_STIR: grid '{}' is empty (above the diagonal)", labels[grid_num]));
      if (lower_sum != 0.)
        warning(format("conv_SimSET_crystal_pairs_to_STIR: grid '{}' has {} on or below the diagonal, which is ignored",
                       labels[grid_num],
                       lower_sum));

      ProjDataInMemory proj_data(template_sptr->get_exam_info_sptr(), proj_data_info_sptr);
      rebin_grid(proj_data, *proj_data_info_ptr, grid, num_crystals, crystal_index);

      const std::string output_filename = output_prefix + "_" + labels[grid_num] + ".hs";
      info(format("conv_SimSET_crystal_pairs_to_STIR: grid '{}': total above the diagonal {}, in the output {}. Writing {}",
                  labels[grid_num],
                  upper_sum,
                  proj_data.sum(),
                  output_filename));
      if (proj_data.write_to_file(output_filename) != Succeeded::yes)
        error(format("conv_SimSET_crystal_pairs_to_STIR: error writing {}", output_filename));
    }

  return EXIT_SUCCESS;
}
