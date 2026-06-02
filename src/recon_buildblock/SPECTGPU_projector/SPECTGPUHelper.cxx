//
//
/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief non-inline implementations for stir::SPECTGPUHelper

  \author Daniel Deidda


*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUHelper.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/is_null_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInMemory.h"
#include "stir/IndexRange3D.h"
#include "stir/FilePath.h"
#include "stir/IO/stir_ecat_common.h"
#include "stir/error.h"
#include "stir/format.h"
#include "stir/cuda_utilities.h"
// Non-STIR includes
#include <fstream>
#include <math.h>
#include "driver_types.h"
// SPECTGPU includes

START_NAMESPACE_STIR

SPECTGPUHelper::~SPECTGPUHelper()
{}


//static shared_ptr<Cnst>
//get_cnst(const Scanner& scanner, const bool cuda_verbose, const char cuda_device)
//{
//  shared_ptr<Cnst> cnt_sptr = MAKE_SHARED<Cnst>();

//  cnt_sptr->DEVID = cuda_device; // device (GPU) ID.  allows choosing the device on which to perform calculations
//  cnt_sptr->VERBOSE = cuda_verbose;
     
//  cnt_sptr->A = NSANGLES; // sino angles
//  cnt_sptr->W = NSBINS;   // sino bins for any angular index
//  cnt_sptr->aw = AW;      // sino bins (active only)

//  cnt_sptr->NCRS = nCRS;   // number of crystals
//  cnt_sptr->NRNG = NRINGS; // number of axial positions
//  cnt_sptr->D = -1;        // number of linear indexes along Michelogram diagonals                         /*unknown*/
//  cnt_sptr->Bt = -1;       // number of buckets transaxially                                               /*unknown*/

//  cnt_sptr->B = NBUCKTS; // number of buckets (total)
//  cnt_sptr->Cbt = 32552; // number of crystals in bucket transaxially                                /*unknown*/
//  cnt_sptr->Cba = 3;     // number of crystals in bucket axially                                         /*unknown*/

//  cnt_sptr->NSN1 = NSINOS;           // number of sinos
//  cnt_sptr->NSN64 = NRINGS * NRINGS; // with no MRD limit
//  cnt_sptr->NSEG0 = SEG0;

//  cnt_sptr->RNG_STRT = 0;
//  cnt_sptr->RNG_END = NRINGS;

//  cnt_sptr->ALPHA = aLPHA;  // angle subtended by a crystal
//  float R = 32.8f;          // ring radius
//  cnt_sptr->RE = R + 0.67f; // effective ring radius accounting for the depth of interaction
//  cnt_sptr->AXR = SZ_RING;  // axial crystal dim

//  float CLGHT = 29979245800.f;                   // speed of light [cm/s]
//  return cnt_sptr;
//}



void
SPECTGPUHelper::set_up()
{
  if (_att < 0)
    throw std::runtime_error("SPECTGPUHelper::set_up() "
                             "emission or transmission mode (att) not set.");

//  // Get consts
//  _cnt_sptr = get_cnst(_scanner_type, _verbose, _devid);


//  // isub
//  _isub = std::vector<int>(unsigned(AW));
//  for (unsigned i = 0; i < unsigned(AW); i++)
//    _isub[i] = int(i);

  _already_set_up = true;
}

void
SPECTGPUHelper::check_set_up() const
{
  if (!_already_set_up)
    throw std::runtime_error("SPECTGPUHelper::check_set_up() "
                             "Make sure filenames have been set and set_up has been run.");
}

std::vector<float>
SPECTGPUHelper::create_SPECTGPU_image()
{
  return std::vector<float>(SZ_IMZ * SZ_IMX * SZ_IMY, 0);
}

shared_ptr<VoxelsOnCartesianGrid<float>>
SPECTGPUHelper::create_stir_im()
{
  int nz(SZ_IMZ), nx(SZ_IMX), ny(SZ_IMY);
  float sz(SZ_VOXZ * 10.f), sx(SZ_VOXY * 10.f), sy(SZ_VOXY * 10.f);
  shared_ptr<VoxelsOnCartesianGrid<float>> out_im_stir_sptr = MAKE_SHARED<VoxelsOnCartesianGrid<float>>(
      IndexRange3D(0, nz - 1, -(ny / 2), -(ny / 2) + ny - 1, -(nx / 2), -(nx / 2) + nx - 1),
      CartesianCoordinate3D<float>(0.f, 0.f, 0.f),
      CartesianCoordinate3D<float>(sz, sy, sx));
  return out_im_stir_sptr;
}

std::vector<float>
SPECTGPUHelper::create_SPECTGPU_sinogram() const
{
  check_set_up();
  return std::vector<float>(_isub.size() * static_cast<unsigned long>(_nsinos), 0);
}


void
get_stir_indices_and_dims(int stir_dim[3],
                          Coordinate3D<int>& min_indices,
                          Coordinate3D<int>& max_indices,
                          const DiscretisedDensity<3, float>& stir)
{
  if (!stir.get_regular_range(min_indices, max_indices))
    throw std::runtime_error("SPECTGPUHelper::set_input - "
                             "expected image to have regular range.");
  for (int i = 0; i < 3; ++i)
    stir_dim[i] = max_indices[i + 1] - min_indices[i + 1] + 1;
}

unsigned
convert_SPECTGPU_im_3d_to_1d_idx(const unsigned x, const unsigned y, const unsigned z)
{
  return z * SZ_IMX * SZ_IMY + y * SZ_IMX + x;
}

unsigned
SPECTGPUHelper::convert_SPECTGPU_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const
{
  return sino * NSANGLES * NSBINS + ang * NSBINS + bins;
}

void

void
SPECTGPUHelper::back_project(std::vector<float>& image, const std::vector<float>& sino_no_gaps) const
{
  check_set_up();
  assert(!image.empty());

  std::vector<float> unpermuted_image = this->create_SPECTGPU_image();

  if (_verbose)
    getMemUse();

  // here start our code that calls the kernels
  //rotation kernel 
  //backward kernel
  
}

void
SPECTGPUHelper::forward_project(Array<3, elemT>& sino, const Array<3, elemT>& image) const
{
  check_set_up();
  assert(!sino.empty());
//  prjdatainmemory
  cudaMalloc(&this->cuda_image, stir_image_sptr->size_all() * sizeof(elemT));
  array_to_device(this->cuda_image, *stir_image_sptr);
  // Permute the data (as this is done on the SPECTGPU python side before forward projection
  // unsigned output_dims[3] = { 320, 320, 127 };
  // unsigned permute_order[3] = { 1, 2, 0 };
  // std::vector<float> permuted_image = this->create_SPECTGPU_image();
  // this->permute(permuted_image, image, output_dims, permute_order);

  std::vector<float> image = this->create_SPECTGPU_image();

  if (_verbose)
    getMemUse();

  // here we do our calculation
  // for view 
  // call rotation kernel
  // call forward kernel
}

shared_ptr<ProjData>
SPECTGPUHelper::create_stir_sino()
{
  
  shared_ptr<ExamInfo> ei_sptr = MAKE_SHARED<ExamInfo>();
  ei_sptr->imaging_modality = ImagingModality::NM;
  shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name("uncknown"));
  int num_views = scanner_sptr->get_num_detectors_per_ring() / 2 ;
  int num_tang_pos = scanner_sptr->get_max_num_non_arccorrected_bins();
  shared_ptr<ProjDataInfo> pdi_sptr
      = ProjDataInfo::construct_proj_data_info(scanner_sptr, span, max_ring_diff, num_views, num_tang_pos, false);
  shared_ptr<ProjDataInMemory> pd_sptr = MAKE_SHARED<ProjDataInMemory>(ei_sptr, pdi_sptr);
  return pd_sptr;
}

template <class dataType>
static dataType*
read_from_binary_file(std::ifstream& file, const unsigned long num_elements)
{
  // Get current position, get size to end and go back to current position
  const unsigned long current_pos = file.tellg();
  file.seekg(std::ios::cur, std::ios::end);
  const unsigned long remaining_elements = file.tellg() / sizeof(dataType);
  file.seekg(current_pos, std::ios::beg);

  if (remaining_elements < num_elements)
    throw std::runtime_error("File smaller than requested.");

  dataType* contents = create_heap_array<dataType>(num_elements);
  file.read(reinterpret_cast<char*>(contents), num_elements * sizeof(dataType));
  return contents;
}


void
check_im_sizes(const int stir_dim[3], const int np_dim[3])
{
  for (int i = 0; i < 3; ++i)
    if (stir_dim[i] != np_dim[i])
      throw std::runtime_error(format("SPECTGPUHelper::check_im_sizes() - "
                                      "STIR image ({}, {}, {}) should be == ({},{},{}).",
                                      stir_dim[0],
                                      stir_dim[1],
                                      stir_dim[2],
                                      np_dim[0],
                                      np_dim[1],
                                      np_dim[2]));
}

void
check_voxel_spacing(const DiscretisedDensity<3, float>& stir)
{
  // Requires image to be a VoxelsOnCartesianGrid
  const VoxelsOnCartesianGrid<float>& stir_vocg = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(stir);
  const BasicCoordinate<3, float> stir_spacing = stir_vocg.get_grid_spacing();

  // Get SPECTGPU image spacing (need to *10 for mm)
  float np_spacing[3] = { 10.f * SZ_VOXZ, 10.f * SZ_VOXY, 10.f * SZ_VOXY };

  for (unsigned i = 0; i < 3; ++i)
    if (std::abs(stir_spacing[int(i) + 1] - np_spacing[i]) > 1e-4f)
      throw std::runtime_error(format("SPECTGPUHelper::check_voxel_spacing() - "
                                      "STIR image ({}, {}, {}) should be == ({},{},{}).",
                                      stir_spacing[1],
                                      stir_spacing[2],
                                      stir_spacing[3],
                                      np_spacing[0],
                                      np_spacing[1],
                                      np_spacing[2]));
}

void
SPECTGPUHelper::convert_image_stir_to_SPECTGPU(std::vector<float>& np_vec, const DiscretisedDensity<3, float>& stir)
{
  // Get the dimensions of the input image
  Coordinate3D<int> min_indices;
  Coordinate3D<int> max_indices;
  int stir_dim[3];
  get_stir_indices_and_dims(stir_dim, min_indices, max_indices, stir);

  // SPECTGPU requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
  // which at the time of writing was (127,320,320).
  const int np_dim[3] = { SZ_IMZ, SZ_IMX, SZ_IMY };
  check_im_sizes(stir_dim, np_dim);
  check_voxel_spacing(stir);

  // Copy data from STIR to SPECTGPU image
  unsigned np_z, np_y, np_x, np_1d;
  for (int z = min_indices[1]; z <= max_indices[1]; z++)
    {
      for (int y = min_indices[2]; y <= max_indices[2]; y++)
        {
          for (int x = min_indices[3]; x <= max_indices[3]; x++)
            {
              // Convert the stir 3d index to a SPECTGPU 1d index
              np_z = unsigned(z - min_indices[1]);
              np_y = unsigned(y - min_indices[2]);
              np_x = unsigned(x - min_indices[3]);
              np_1d = convert_SPECTGPU_im_3d_to_1d_idx(np_x, np_y, np_z);
              np_vec[np_1d] = stir[z][y][x];
            }
        }
    }
}

void
SPECTGPUHelper::convert_image_SPECTGPU_to_stir(DiscretisedDensity<3, float>& stir, const std::vector<float>& np_vec)
{
  // Get the dimensions of the input image
  Coordinate3D<int> min_indices;
  Coordinate3D<int> max_indices;
  int stir_dim[3];
  get_stir_indices_and_dims(stir_dim, min_indices, max_indices, stir);

  // SPECTGPU requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
  // which at the time of writing was (127,320,320).
  const int np_dim[3] = { SZ_IMZ, SZ_IMX, SZ_IMY };
  check_im_sizes(stir_dim, np_dim);
  check_voxel_spacing(stir);

  // Copy data from SPECTGPU to STIR image
  unsigned np_z, np_y, np_x, np_1d;
  for (int z = min_indices[1]; z <= max_indices[1]; z++)
    {
      for (int y = min_indices[2]; y <= max_indices[2]; y++)
        {
          for (int x = min_indices[3]; x <= max_indices[3]; x++)
            {
              // Convert the stir 3d index to a SPECTGPU 1d index
              np_z = unsigned(z - min_indices[1]);
              np_y = unsigned(y - min_indices[2]);
              np_x = unsigned(x - min_indices[3]);
              np_1d = convert_SPECTGPU_im_3d_to_1d_idx(np_x, np_y, np_z);
              stir[z][y][x] = np_vec[np_1d];
            }
        }
    }
}

void
get_vals_for_proj_data_conversion(std::vector<int>& sizes,
                                  std::vector<int>& segment_sequence,
                                  int& num_sinograms,
                                  int& min_view,
                                  int& max_view,
                                  int& min_tang_pos,
                                  int& max_tang_pos,
                                  const ProjDataInfo& proj_data_info,
                                  const std::vector<float>& np_vec)
{
  const ProjDataInfoCylindricalNoArcCorr* info_sptr = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(&proj_data_info);
  if (is_null_ptr(info_sptr))
    error("SPECTGPUHelper: only works with cylindrical projection data without arc-correction");

  segment_sequence = ecat::find_segment_sequence(proj_data_info);
  sizes.resize(segment_sequence.size());
  for (std::size_t s = 0U; s < segment_sequence.size(); ++s)
    sizes[s] = proj_data_info.get_num_axial_poss(segment_sequence[s]);

  // Get dimensions of STIR sinogram
  min_view = proj_data_info.get_min_view_num();
  max_view = proj_data_info.get_max_view_num();
  min_tang_pos = proj_data_info.get_min_tangential_pos_num();
  max_tang_pos = proj_data_info.get_max_tangential_pos_num();

  num_sinograms = proj_data_info.get_num_axial_poss(0);
  for (int s = 1; s <= proj_data_info.get_max_segment_num(); ++s)
    num_sinograms += 2 * proj_data_info.get_num_axial_poss(s);

  int num_proj_data_elems = num_sinograms * (1 + max_view - min_view) * (1 + max_tang_pos - min_tang_pos);

  // Make sure they're the same size
  if (np_vec.size() != unsigned(num_proj_data_elems))
    error(format("SPECTGPUHelper::get_vals_for_proj_data_conversion "
                 "SPECTGPU and STIR sinograms are different sizes ({} for STIR versus {} for NP",
                 num_proj_data_elems,
                 np_vec.size()));
}

void
get_stir_segment_and_axial_pos_from_SPECTGPU_sino(
    int& segment, int& axial_pos, const unsigned np_sino, const std::vector<int>& sizes, const std::vector<int>& segment_sequence)
{
  int z = int(np_sino);
  for (unsigned i = 0; i < segment_sequence.size(); ++i)
    {
      if (z < sizes[i])
        {
          axial_pos = z;
          segment = segment_sequence[i];
          return;
        }
      else
        {
          z -= sizes[i];
        }
    }
}

void
get_SPECTGPU_sino_from_stir_segment_and_axial_pos(unsigned& np_sino,
                                                  const int segment,
                                                  const int axial_pos,
                                                  const std::vector<int>& sizes,
                                                  const std::vector<int>& segment_sequence)
{
  np_sino = 0U;
  for (unsigned i = 0; i < segment_sequence.size(); ++i)
    {
      if (segment == segment_sequence[i])
        {
          np_sino += axial_pos;
          return;
        }
      else
        {
          np_sino += sizes[i];
        }
    }
  throw std::runtime_error(
      "SPECTGPUHelper::get_SPECTGPU_sino_from_stir_segment_and_axial_pos(): Failed to find SPECTGPU sinogram.");
}

void
SPECTGPUHelper::convert_viewgram_stir_to_SPECTGPU(std::vector<float>& np_vec, const Viewgram<float>& viewgram) const
{
  // Get the values (and LUT) to be able to switch between STIR and SPECTGPU projDatas
  std::vector<int> sizes, segment_sequence;
  int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
  get_vals_for_proj_data_conversion(sizes,
                                    segment_sequence,
                                    num_sinograms,
                                    min_view,
                                    max_view,
                                    min_tang_pos,
                                    max_tang_pos,
                                    *viewgram.get_proj_data_info_sptr(),
                                    np_vec);

  const int segment = viewgram.get_segment_num();
  const int view = viewgram.get_view_num();

  // Loop over the STIR view and tangential position
  for (int ax_pos = viewgram.get_min_axial_pos_num(); ax_pos <= viewgram.get_max_axial_pos_num(); ++ax_pos)
    {

      unsigned np_sino;

      // Convert the SPECTGPU sinogram to STIR's segment and axial position
      get_SPECTGPU_sino_from_stir_segment_and_axial_pos(np_sino, segment, ax_pos, sizes, segment_sequence);

      for (int tang_pos = min_tang_pos; tang_pos <= max_tang_pos; ++tang_pos)
        {

          unsigned np_ang = unsigned(view - min_view);
          unsigned np_bin = unsigned(tang_pos - min_tang_pos);
          unsigned np_1d = convert_SPECTGPU_proj_3d_to_1d_idx(np_ang, np_bin, np_sino);
          np_vec.at(np_1d) = viewgram.at(ax_pos).at(tang_pos);
        }
    }
}

void
SPECTGPUHelper::convert_proj_data_stir_to_SPECTGPU(std::vector<float>& np_vec, const ProjData& stir) const
{
  const int min_view = stir.get_min_view_num();
  const int max_view = stir.get_max_view_num();
  const int min_segment = stir.get_min_segment_num();
  const int max_segment = stir.get_max_segment_num();

  for (int view = min_view; view <= max_view; ++view)
    {
      for (int segment = min_segment; segment <= max_segment; ++segment)
        {
          convert_viewgram_stir_to_SPECTGPU(np_vec, stir.get_viewgram(view, segment));
        }
    }
}

void
SPECTGPUHelper::convert_proj_data_SPECTGPU_to_stir(ProjData& stir, const std::vector<float>& np_vec) const
{
  // Get the values (and LUT) to be able to switch between STIR and SPECTGPU projDatas
  std::vector<int> sizes, segment_sequence;
  int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
  get_vals_for_proj_data_conversion(sizes,
                                    segment_sequence,
                                    num_sinograms,
                                    min_view,
                                    max_view,
                                    min_tang_pos,
                                    max_tang_pos,
                                    *stir.get_proj_data_info_sptr(),
                                    np_vec);

  int segment, axial_pos;
  // Loop over all SPECTGPU sinograms
  for (unsigned np_sino = 0; np_sino < unsigned(num_sinograms); ++np_sino)
    {

      // Convert the SPECTGPU sinogram to STIR's segment and axial position
      get_stir_segment_and_axial_pos_from_SPECTGPU_sino(segment, axial_pos, np_sino, sizes, segment_sequence);

      // Get the corresponding STIR sinogram
      Sinogram<float> sino = stir.get_empty_sinogram(axial_pos, segment);

      // Loop over the STIR view and tangential position
      for (int view = min_view; view <= max_view; ++view)
        {
          for (int tang_pos = min_tang_pos; tang_pos <= max_tang_pos; ++tang_pos)
            {

              unsigned np_ang = unsigned(view - min_view);
              unsigned np_bin = unsigned(tang_pos - min_tang_pos);
              unsigned np_1d = convert_SPECTGPU_proj_3d_to_1d_idx(np_ang, np_bin, np_sino);
              sino.at(view).at(tang_pos) = np_vec.at(np_1d);
            }
        }
      stir.set_sinogram(sino);
    }
}

END_NAMESPACE_STIR
