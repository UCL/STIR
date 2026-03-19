/* CListModeDataPETSIRD.cxx

Coincidence LM Data Class for PETSIRD: Implementation

    Copyright 2025, UMCG
    Copyright 2025, MGH / HST A. Martinos Center for Biomedical Imaging

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

             http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
*/
/*!

\file
\ingroup listmode
\brief implementation of class stir::CListModeDataPETSIRD

\author Nikos Efthimiou
*/

#include "stir/PETSIRDInfo.h"
#include "stir/detail/PETSIRDInfo_helpers.h"
#include "stir/Succeeded.h"

#include "stir/warning.h"

#include "petsird_helpers.h"
#include "petsird_helpers/create.h"
#include "petsird_helpers/geometry.h"

#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"

START_NAMESPACE_STIR

/*!
  \brief Infer scanner blocks and rotation axis from PETSIRD replicated modules.
  \param unique_dim1_values Output set of unique translations along X.
  \param unique_dim2_values Output set of unique translations along Y.
  \param unique_dim3_values Output set of unique translations along Z.
  \param replicated_module_list PETSIRD replicated detector modules.
  \return Index of rotation axis (0=x, 1=y, 2=z) or -1 if not found.

  \details
  - Extracts translation components into the provided sets.
  - Uses skew-symmetric part of rotation matrices to infer axis direction.
  - Emits warnings for mixed-axis rotations or inconsistencies.
*/
int
PETSIRDInfo::figure_out_scanner_blocks_and_rotation_axis(std::set<float>& unique_dim1_values,
                                                         std::set<float>& unique_dim2_values,
                                                         std::set<float>& unique_dim3_values)
{
  auto insertTranslations = [&](const petsird::RigidTransformation& trans) {
    unique_dim1_values.insert(trans.matrix.at(0, 3));
    unique_dim2_values.insert(trans.matrix.at(1, 3));
    unique_dim3_values.insert(trans.matrix.at(2, 3));
  };

  auto extractRotationMatrix = [](const petsird::RigidTransformation& trans) -> matrix::Mat3 {
    matrix::Mat3 R;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        R[i][j] = trans.matrix.at(i, j);
    return R;
    // skew = matrix::subtract(R, matrix::transpose(R));
    // auto rot = matrix::getAxisFromSkew(skew);
  };

  std::array<std::array<float, 3>, 3> skew;
  int detected_axis = -1;

  std::vector<petsird::ReplicatedDetectorModule> replicated_module_list
      = petsird_scanner_info_sptr->scanner_geometry.replicated_modules;
  for (const auto& module : replicated_module_list)
    for (const auto& mod_trans : module.transforms)
      {
        insertTranslations(mod_trans);
        matrix::Mat3 R = extractRotationMatrix(mod_trans);
        skew = matrix::subtract(R, matrix::transpose(R));
        auto axis_vec = matrix::getAxisFromSkew(skew);

        int current_axis = -1;
        for (int i = 0; i < 3; ++i)
          {
            if (std::abs(axis_vec[i]) > 1e-6f)
              {
                if (current_axis != -1)
                  {
                    warning("Rotation involves multiple axis components. Possibly non-pure rotation.");
                    current_axis = -2; // Sentinel for mixed axes
                    return -1;
                  }
                current_axis = i;
              }
          }

        if (current_axis >= 0)
          {
            if (detected_axis == -1)
              detected_axis = current_axis;
            else if (detected_axis != current_axis)
              warning("Inconsistent rotation axis detected between modules.");
          }
      }
  info(format("Rotation axis of blocks inferred as axis index {}", detected_axis));
  return detected_axis;
}

/*!
  \brief Determine block element translations and radius.
  \param unique_dim1_values Output set of element translations along X.
  \param unique_dim2_values Output set of element translations along Y.
  \param unique_dim3_values Output set of element translations along Z.
  \param radius Output inferred radius (positive component orthogonal to rotation axis).
  \param radius_index Output index of the axis containing the radius component.
  \param rotation_axis Known rotation axis (0=x, 1=y, 2=z).
  \param replicated_module_list PETSIRD replicated detector modules.

  \details
  - Scans element-level transforms to infer radius and collect translations.
  - Emits warnings if mixed radii are detected.
*/
void
PETSIRDInfo::figure_out_block_element_transformations(std::set<float>& unique_dim1_values,
                                                      std::set<float>& unique_dim2_values,
                                                      std::set<float>& unique_dim3_values,
                                                      float& radius,
                                                      int& radius_index,
                                                      const int rotation_axis)
{
  auto insert_translation = [&](const petsird::RigidTransformation& trans) {
    unique_dim1_values.insert(trans.matrix.at(0, 3));
    unique_dim2_values.insert(trans.matrix.at(1, 3));
    unique_dim3_values.insert(trans.matrix.at(2, 3));
  };

  auto detect_radius = [&](const petsird::RigidTransformation& trans) -> bool {
    for (int i = 0; i < 3; ++i)
      {
        if (i == rotation_axis)
          continue;
        float candidate = trans.matrix.at(i, 3);
        if (candidate > 0.0f)
          {
            radius = candidate;
            radius_index = i;
            return true;
          }
      }
    return false;
  };

  for (const auto& module : petsird_scanner_info_sptr->scanner_geometry.replicated_modules)
    {
      for (const auto& el_trans : module.object.detecting_elements.transforms)
        {
          if (radius == 0.0f)
            {
              if (!detect_radius(el_trans))
                {
                  error("Unable to determine radius from translation components.");
                }
            }
          else
            {
              float current = el_trans.matrix.at(radius_index, 3);
              if (std::abs(current - radius) > 1e-4f)
                warning("Mixed radii detected. Consider checking for misaligned modules.");
            }

          insert_translation(el_trans);
          // std::cout << el_trans.matrix << std::endl;
        }
    }
}

void
PETSIRDInfo::figure_out_block_angles(std::set<float>& unique_angle_modules, const int rot_axis)
{

  for (const auto& module : petsird_scanner_info_sptr->scanner_geometry.replicated_modules)
    for (const auto& transform : module.transforms)
      {
        if (rot_axis == 0)
          unique_angle_modules.insert(
              std::fabs(int(1000.F * std::atan2(transform.matrix.at(1, 0), transform.matrix.at(2, 0))) / 1000.F));
        else if (rot_axis == 1)
          unique_angle_modules.insert(
              std::fabs(int(1000.F * std::atan2(transform.matrix.at(2, 0), transform.matrix.at(0, 0))) / 1000.F));
        else if (rot_axis == 2)
          unique_angle_modules.insert(
              std::fabs(int(1000.F * std::atan2(transform.matrix.at(1, 0), transform.matrix.at(0, 0))) / 1000.F));
      }
}

PETSIRDInfo::PETSIRDInfo(const petsird::Header& header, std::string scanner_geometry)
    : petsird_scanner_info_sptr(std::make_shared<const petsird::ScannerInformation>(header.scanner)),
      petsird_header_sptr(std::make_shared<const petsird::Header>(header)),
      forced_geometry(scanner_geometry)
{

  if (!petsird_scanner_info_sptr)
    error("PETSIRDInfo: Null PETSIRD ScannerInformation pointer provided.");

  const auto& geom = petsird_scanner_info_sptr->scanner_geometry;

  if (geom.replicated_modules.empty())
    error("PETSIRDInfo: scanner_geometry.replicated_modules is empty.");

  if (geom.replicated_modules[0].transforms.empty())
    warning("PETSIRDInfo: replicated_modules[0].transforms is empty (rotation/angles may be unreliable).");

  if (geom.replicated_modules[0].object.detecting_elements.transforms.empty())
    error("PETSIRDInfo: detecting_elements.transforms is empty (cannot infer element spacing/radius).");

  //! TODO: Determine the DOI based on material
  float average_doi = 0.0;
  if (petsird_scanner_info_sptr->bulk_materials.size() > 0)
    {
      const std::string& material = petsird_scanner_info_sptr->bulk_materials[0].name;
      if (material.size() > 0)
        average_doi = (material == "BGO") ? 5.0f : (material == "LSO" || material == "LYSO") ? 7.0f : 0.0f;
    }

  type_of_module = petsird_scanner_info_sptr->scanner_geometry.replicated_modules.size() - 1;
  if (type_of_module > 0)
    {
      error("Multiple types of PETSIRD modules are not supported. Abort.");
    }

  module_pair = petsird::TypeOfModulePair{ type_of_module, type_of_module };

  const auto& tof_bin_edges = petsird_scanner_info_sptr->tof_bin_edges[type_of_module][type_of_module];
  info(fmt::format("Num. of TOF bins in PETSIRD {}", tof_bin_edges.NumberOfBins()));
  if (tof_bin_edges.NumberOfBins() > 1)
    {
      info(fmt::format(
          "Since the PETSIRD file has TOF information, STIR will force cylindrical geometry, as long as other things checkout."));
      forced_geometry = "cylindrical";
    }

  std::set<float> unique_tof_values;
  vector_utils::find_unique_values_2D(unique_tof_values, petsird_scanner_info_sptr->tof_resolution);
  //! TODO: Supports only single type of module
  numberOfModules = petsird_scanner_info_sptr->scanner_geometry.replicated_modules[0].NumberOfObjects();
  //! TODO: Supports only single type of module
  numberOfElementsIndices
      = petsird_scanner_info_sptr->scanner_geometry.replicated_modules[0].object.detecting_elements.NumberOfObjects();
  std::set<float> unique_dim1_values, unique_dim2_values, unique_dim3_values;
  std::set<float> unique_tof_resolutions;

  const int rotation_axis
      = figure_out_scanner_blocks_and_rotation_axis(unique_dim1_values, unique_dim2_values, unique_dim3_values);
  if (rotation_axis == -1)
    is_cylindrical = false;

  const std::set<float>& main_axis = vector_utils::get_largest_vector(unique_dim1_values, unique_dim2_values, unique_dim3_values);

  int num_transaxial_blocks = numberOfModules / main_axis.size();
  info(fmt::format("I deduce that the scanner has {} transaxial number of blocks", num_transaxial_blocks));

  float radius = 0;
  int radius_indx = -1;

  std::set<float> unique_elements_dim1_values, unique_elements_dim2_values, unique_elements_dim3_values;
  figure_out_block_element_transformations(
      unique_elements_dim1_values, unique_elements_dim2_values, unique_elements_dim3_values, radius, radius_indx, rotation_axis);

  std::set<float> unique_angle_modules;
  figure_out_block_angles(unique_angle_modules, rotation_axis);

  std::vector<float> block_angular_spacing;
  if (!vector_utils::get_spacing_uniform(
          block_angular_spacing, unique_angle_modules, 1e-2)) /// epsilon * 10000) // relax epsilon here
    {
      is_cylindrical = false;
    }

  std::size_t group2 = 0, group3 = 0;
  {
    std::vector<CartesianCoordinate3D<float>> pet_sird_positions;

    for (uint32_t module = 0; module < 1; module++)
      {
        for (uint32_t elem = 0; elem < numberOfElementsIndices; elem++)
          {
            petsird::ExpandedDetectionBin expanded_detection_bin{ module, elem, 1 };
            auto box_shape = petsird_helpers::geometry::get_detecting_box(
                *petsird_scanner_info_sptr, type_of_module, expanded_detection_bin);
            CartesianCoordinate3D<float> mean_coord;
            for (auto& corner : box_shape.corners)
              { // if STIR (z,y,x) -> PETSIRD (-y, -x, z) pheraps  the order below needs to be changed
                mean_coord.x() = +corner.c[0] / box_shape.corners.size();
                mean_coord.y() = +corner.c[1] / box_shape.corners.size();
                mean_coord.z() = +corner.c[2] / box_shape.corners.size();
              }
            // mean_coord.z() += this_scanner_sptr->get_axial_crystal_spacing() /
            //       save  mean pos into map
            pet_sird_positions.push_back(mean_coord);
          }
      }

    if (inferGroupSizes_dim2_dim3(pet_sird_positions, group2, group3))
      {
        std::cout << "groupSize_dim2 = " << group2 << "\n";

        std::cout << "groupSize_dim3 = " << group3 << "\n";
      }
    else
      {
        std::cout << "No (dim2, dim3) loop structure detected.\n";
        group2 = 1;
        group3 = 1;
      }
  }

  std::vector<float> element_horizontal_spacing, element_vertical_spacing;
  std::set<float> unique_elements_horizontal_values, unique_elements_vertical_values;
  if (radius_indx == 0)
    {
      if (!vector_utils::get_spacing_uniform(element_horizontal_spacing, unique_elements_dim3_values))
        {
          is_cylindrical = false;
        }
      if (!vector_utils::get_spacing_uniform(element_vertical_spacing, unique_elements_dim2_values))
        {
          is_cylindrical = false;
        }
      unique_elements_horizontal_values = unique_elements_dim3_values;
      unique_elements_vertical_values = unique_elements_dim2_values;
    }
  else
    {
      //! TODO: Multiple radii handling
      error("Multiple radii handling not implemented yet.");
    }

  {
    info("Printing TOF bin edges for validation (please make sure that the STIR TOF bin edges match the PETSIRD TOF bin edges):");
    for (size_t i = 0; i < tof_bin_edges.NumberOfBins(); ++i)
      {
        std::cout << "PETSIRD TOF bin edge " << i << ": " << tof_bin_edges.edges[i] << "\n";
      }
  }

  bool has_tile_structure = group2 > 1 && group3 > 1 && (numberOfElementsIndices % (group2 * group3) == 0);
  info(fmt::format("Has tile structure: {}", has_tile_structure ? "yes" : "no"));

  if (has_tile_structure)
    {
      // GATE-style tiled PETSIRD
      blocks_per_bucket_transaxial = group2 > 1 ? unique_elements_vertical_values.size() / group2 : group2;
      std::cout << "blocks per bucket in transaxial direction = " << blocks_per_bucket_transaxial << "\n";
      blocks_per_bucket_axial
          = group3 > 1 ? unique_elements_horizontal_values.size() / (numberOfElementsIndices / group3) : group3;
      std::cout << "blocks per bucket in axial direction =  " << blocks_per_bucket_axial << "\n";
      num_axial_crystals_per_block = unique_elements_horizontal_values.size() / blocks_per_bucket_axial;
      num_trans_crystals_per_block = unique_elements_vertical_values.size() / blocks_per_bucket_transaxial;
    }
  else
    {
      // STIR-style flattened PETSIRD
      blocks_per_bucket_transaxial = 1;
      blocks_per_bucket_axial = 1;

      num_trans_crystals_per_block = unique_elements_vertical_values.size();
      num_axial_crystals_per_block = unique_elements_horizontal_values.size();

      warning("No block structure detected: falling back to flat crystal layout.");
    }

  std::vector<float> block_axial_spacing;
  vector_utils::get_spacing_uniform(block_axial_spacing, main_axis);
  if (block_axial_spacing.size() < 1)
    {
      //      std::set<int>::iterator it = unique_elements_horizontal_values.begin();
      //      std::advance(it,0);
      float begin = *std::next(unique_elements_horizontal_values.begin(), 0);
      //      std::advance(it,unique_elements_horizontal_values.size()-1);
      float end = *std::next(unique_elements_horizontal_values.begin(), unique_elements_horizontal_values.size() - 1);
      block_axial_spacing.push_back(std::abs(end - begin));
    }

  info(fmt::format("I counted {} axial blocks with spacing {}", unique_dim3_values.size(), block_axial_spacing[0]));
  // Check if the cyrcle area is less than 5% different from the polygon
  float expected_circle_area = float(M_PI * radius * radius);
  float polygon_area
      = float(0.5f * unique_angle_modules.size() * radius * radius * std::sin(2.f * float(M_PI) / unique_angle_modules.size()));
  info(fmt::format("Circle area: {}, Polygon area: {}, pct {}",
                   expected_circle_area,
                   polygon_area,
                   std::abs(expected_circle_area - polygon_area) / expected_circle_area));
  // This is the NeuroLF ratio.
  forced_geometry = "BlocksOnCylindrical"; 
  if (std::abs(expected_circle_area - polygon_area) / expected_circle_area < 0.02f || forced_geometry == "cylindrical")
    {
      info(fmt::format("PETSIRDInfo: The cylindrical area {} is more than 95% matching the polygon area {}. We will presume a "
                       "cylindrical configuration.",
                       expected_circle_area,
                       polygon_area));
      stir_scanner_sptr.reset(
          new Scanner(Scanner::User_defined_scanner,
                      std::string("PETSIRD_defined_scanner"),
                      /* num dets per ring */
                      (num_transaxial_blocks * unique_elements_vertical_values.size()),
                      unique_dim3_values.size() * unique_elements_horizontal_values.size() /* num of rings */,
                      /* number of non arccor bins */
                      (num_transaxial_blocks * unique_elements_vertical_values.size()) / 2,
                      /* number of maximum arccor bins */
                      (num_transaxial_blocks * unique_elements_vertical_values.size()) / 2,
                      /* inner ring radius */
                      radius,
                      /* doi */ average_doi, // average_doi,
                      /* ring spacing */
                      element_horizontal_spacing[0], //* 10.f,
                      // bin_size_v
                      radius < 140 ? element_vertical_spacing[0] / 2 : element_vertical_spacing[0], // * 10.f,, // * 10.f,
                      /*intrinsic_tilt_v*/
                      0.f,
                      /*num_axial_blocks_per_bucket_v */
                      blocks_per_bucket_axial,
                      /*num_transaxial_blocks_per_bucket_v*/
                      blocks_per_bucket_transaxial,
                      /*num_axial_crystals_per_block_v*/
                      num_axial_crystals_per_block,
                      /*num_transaxial_crystals_per_block_v*/
                      num_trans_crystals_per_block,
                      /*num_axial_crystals_per_singles_unit_v*/
                      unique_elements_horizontal_values.size() / blocks_per_bucket_axial,
                      /*num_transaxial_crystals_per_singles_unit_v*/
                      unique_elements_vertical_values.size() / blocks_per_bucket_transaxial,
                      /*num_detector_layers_v*/
                      1,                                                           // num_detector_layers_v
                      petsird_scanner_info_sptr->energy_resolution_at_511.front(), // energy_resolution_v
                      511,                                                         // reference_energy_v
                      tof_bin_edges.NumberOfBins(),
                      (tof_bin_edges.edges[1] - tof_bin_edges.edges[0]) / speed_of_light_in_mm_per_ps_div2,
                      *unique_tof_values.begin() * 10 // non-TOF
                      ));
      is_cylindrical = true;
      is_generic_geometry = false;
      is_block_configuration = false;
    }
  else
    {
      const uint32_t forced_axial_buckets = unique_dim3_values.size();
      info("PETSIRDInfo: The cylindrical area is less than 95% matching the polygon area. We will predsume a non-cylindrical "
           "configuration.");
      stir_scanner_sptr.reset(
          new Scanner(Scanner::User_defined_scanner,
                      std::string("PETSIRD_defined_scanner"),
                      /* num dets per ring */
                      (num_transaxial_blocks * unique_elements_vertical_values.size()),
                      unique_dim3_values.size() * unique_elements_horizontal_values.size() /* num of rings */,
                      /* number of non arccor bins */
                      (num_transaxial_blocks * unique_elements_vertical_values.size()) / 2,
                      /* number of maximum arccor bins */
                      (num_transaxial_blocks * unique_elements_vertical_values.size()) / 2,
                      /* inner ring radius */
                      radius,
                      /* doi */ average_doi,
                      /* ring spacing */
                      element_horizontal_spacing[0], //* 10.f,
                      // bin_size_v
                      radius < 140 ? element_vertical_spacing[0] / 2 : element_vertical_spacing[0], // * 10.f,
                      /*intrinsic_tilt_v*/
                      0.f,
                      /*num_axial_blocks_per_bucket_v */
                      forced_axial_buckets,
                      /*num_transaxial_blocks_per_bucket_v*/
                      1,
                      /*num_axial_crystals_per_block_v*/
                      blocks_per_bucket_axial * num_axial_crystals_per_block,
                      /*num_transaxial_crystals_per_block_v*/
                      blocks_per_bucket_transaxial * num_trans_crystals_per_block,
                      /*num_axial_crystals_per_singles_unit_v*/
                      unique_elements_horizontal_values.size() / blocks_per_bucket_axial,
                      /*num_transaxial_crystals_per_singles_unit_v*/
                      unique_elements_vertical_values.size() / blocks_per_bucket_transaxial,
                      /*num_detector_layers_v*/
                      1,                                                           // num_detector_layers_v
                      petsird_scanner_info_sptr->energy_resolution_at_511.front(), // energy_resolution_v
                      511,                                                         // reference_energy_v
                      1,
                      0.F,
                      0.F,                   // non-TOF
                      "BlocksOnCylindrical", // scanner_geometry_v
                      (*std::next(unique_elements_horizontal_values.begin())
                       - *unique_elements_horizontal_values.begin()), // axial_crystal_spacing_v
                      (*std::next(unique_elements_vertical_values.begin())
                       - *unique_elements_vertical_values.begin()), // transaxial_crystal_spacing_v
                      (*std::next(unique_elements_horizontal_values.begin()) - *unique_elements_horizontal_values.begin())
                          * num_axial_crystals_per_block * blocks_per_bucket_axial, // axial_block_spacing_v
                      (*std::next(unique_elements_vertical_values.begin()) - *unique_elements_vertical_values.begin())
                          * num_trans_crystals_per_block * blocks_per_bucket_transaxial, // transaxial_block_spacing_v
                      ""                                                                 // crystal_map_file_name_v
                      ));
      is_cylindrical = false;
      is_generic_geometry = false;
      is_block_configuration = true;
    }

  /// Now let's create the PETISIRD - STIR geometry mapping
  petsird_to_stir = std::make_shared<PETSIRDToSTIRDetectorIndexMap>();
  petsird_map_sptr = std::make_shared<DetectorCoordinateMap::det_pos_to_coord_type>();

  enum class InnerLoopDim
  {
    Axial,
    Tangential,
    Radial
  };
  InnerLoopDim inner_dim = InnerLoopDim::Tangential; // determined from your groupSize analysis

  // PRECOMPUTED from previous step:
  std::size_t groupSize
      = blocks_per_bucket_transaxial == 1 ? 1 : num_trans_crystals_per_block; // e.g. 5, or 1 if purely monotonic
  // extern InnerLoopDim inner_dim;         // Axial / Tangential / Radial

  // Don't need these anymore. Keeping for future reference.
  // const int num_ax = blocks_per_bucket_axial * num_axial_crystals_per_block;
  // const int num_tang = blocks_per_bucket_transaxial * num_trans_crystals_per_block;

  info(fmt::format("Tile size (groupSize) = {}", groupSize));

  // const uint32_t num_rings = static_cast<uint32_t>(stir_scanner_sptr->get_num_rings());
  // const uint32_t num_det = static_cast<uint32_t>(stir_scanner_sptr->get_num_detectors_per_ring());
  const uint32_t axial_blocks = static_cast<uint32_t>(stir_scanner_sptr->get_num_axial_blocks());
  const uint32_t trans_crys = static_cast<uint32_t>(stir_scanner_sptr->get_num_transaxial_crystals_per_block());
  const uint32_t axial_crys = static_cast<uint32_t>(stir_scanner_sptr->get_num_axial_crystals_per_block());
  // const uint32_t layers = static_cast<uint32_t>(stir_scanner_sptr->get_num_detector_layers());

  for (uint32_t module = 0; module < numberOfModules; ++module)
    for (uint32_t elem = 0; elem < numberOfElementsIndices; ++elem)
      {
        petsird::ExpandedDetectionBin bin{ module, elem, 0 };

        int tang_pos = 0, ax_pos = 0, rad_pos = 0;

        if (!has_tile_structure)
          {
            // -------- STIR-origin PETSIRD (flat) --------
            const uint32_t ax_mod = module % axial_blocks;
            const uint32_t tang_mod = module / axial_blocks;

            const uint32_t axial_in_block = elem % axial_crys;
            const uint32_t tmp = elem / axial_crys;
            const uint32_t trans_in_block = tmp % trans_crys;
            const uint32_t radial = tmp / trans_crys;

            tang_pos = tang_mod * trans_crys + trans_in_block;
            ax_pos = ax_mod * axial_crys + axial_in_block;
            rad_pos = radial;
          }
        else
          {
            // -------- GATE-origin PETSIRD (tiled) --------
            const uint32_t groupSize = num_trans_crystals_per_block;
            const uint32_t tileSize = groupSize * groupSize;

            const uint32_t tile = elem / tileSize;
            const uint32_t inTile = elem % tileSize;

            const uint32_t i0 = inTile % groupSize;
            const uint32_t i1 = inTile / groupSize;

            const uint32_t tang_block = tile % blocks_per_bucket_transaxial;
            const uint32_t axial_block = tile / blocks_per_bucket_transaxial;

            tang_pos = tang_block * groupSize + i0 + module * (num_trans_crystals_per_block * blocks_per_bucket_transaxial);
            ax_pos = axial_block * groupSize + i1;
          }

        (*petsird_to_stir)[bin] = DetectionPosition<>(tang_pos, ax_pos, rad_pos);
      }

  // Reverse the mapping: from STIR detpos to PETSIRD mean coord
  auto map = std::make_shared<STIRToPETSIRDDetectorIndexMap>();

  for (const auto& [petsird_bin, stir_pos] : (*petsird_to_stir))
    {
      auto [it, inserted] = map->emplace(stir_pos, petsird_bin);
      if (!inserted)
        error("Non-unique STIR DetectionPosition while building reverse map");
    }

  stir_to_petsird = map;

  if (petsird_to_stir->size() != stir_to_petsird->size())
    {
      info(fmt::format("PETSIRDInfo: Map size mismatch! Forward size: {0}\n Reverse size: {1}", 
        petsird_to_stir->size(), stir_to_petsird->size()));

      error("Forward and reverse maps differ in size");
    }

  //   for (const auto& [petsird_bin, stir_pos] : *petsird_to_stir)
  // {
  //   auto it = stir_to_petsird->find(stir_pos);

  //   if (it == stir_to_petsird->end())
  //   {
  //     std::cerr << "Missing reverse entry for STIR position: "
  //               << stir_pos.tangential_coord() << ", "
  //               << stir_pos.axial_coord() << ", "
  //               << stir_pos.radial_coord() << std::endl;
  //     error("Reverse map lookup failed");
  //   }

  //   if (!(it->second == petsird_bin))
  //   {
  //     std::cerr << "Mismatch after round-trip!\n"
  //               << "Forward PETSIRD bin: "<<  "\n"
  //               << "Recovered PETSIRD bin: " << "\n"
  //               << "STIR position: "
  //               << stir_pos.tangential_coord() << ", "
  //               << stir_pos.axial_coord() << ", "
  //               << stir_pos.radial_coord() << std::endl;
  //     error("Forward/backward maps are inconsistent");
  //   }
  // }

  // // this->map.reset(new DetectorCoordinateMapLightPETSIRD(petsird_map));
  // this->map.reset(new DetectorCoordinateMap(petsird_map));
  // // this_scanner_sptr->get_detector_map_sptr()->set_detector_coordinate_map_light_sptr(
  // //     std::make_shared<DetectorCoordinateMapLightPETSIRD>(petsird_map));
  // this->map->write_detectormap_to_file("petsird_detector_map_from_scanner_definition.txt");
  // this_scanner_sptr->set_detector_map(petsird_map);
  // this_scanner_sptr->set_up();
}

END_NAMESPACE_STIR