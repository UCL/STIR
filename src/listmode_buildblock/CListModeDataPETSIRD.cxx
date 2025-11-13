/* CListModeDataPETSIRD.cxx

Coincidence LM Data Class for PETSIRD: Implementation

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

\author Daniel Deidda
\author Nikos Efthimiou
*/

#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/error.h"

#include "petsird_helpers.h"
#include "petsird_helpers/create.h"
#include "petsird_helpers/geometry.h"

#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"

#include "stir/listmode/CListModeDataPETSIRD.h"
#include "stir/listmode/CListRecordPETSIRD.h"
#include <set>

START_NAMESPACE_STIR

namespace matrix
{

using Mat3 = std::array<std::array<float, 3>, 3>;
using Vec3 = std::array<float, 3>;

inline Mat3
transpose(const Mat3& mat)
{
  std::array<std::array<float, 3>, 3> result{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      result[j][i] = mat[i][j];
  return result;
}

inline Mat3
subtract(const Mat3& A, const Mat3& B)
{
  std::array<std::array<float, 3>, 3> result{};
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      result[i][j] = A[i][j] - B[i][j];
  return result;
}

inline Vec3
getAxisFromSkew(const Mat3& S)
{
  return {
    0.5f * (S[2][1] - S[1][2]), // x
    0.5f * (S[0][2] - S[2][0]), // y
    0.5f * (S[1][0] - S[0][1])  // z
  };
}

} // namespace matrix

inline bool
get_spacing_uniform(std::vector<float>& spacing, const std::set<float>& unsorted_block_poss, double epsilon = 1e-4)
{
  std::vector<float> sorted_z(unsorted_block_poss.begin(), unsorted_block_poss.end());
  for (size_t i = 1; i < sorted_z.size(); ++i)
    {
      spacing.push_back(std::abs(sorted_z[i] - sorted_z[i - 1]));
    }

  return std::all_of(spacing.begin(), spacing.end(), [&](float s) { return std::abs(s - spacing.front()) <= epsilon; });
}

const std::set<float>&
getLargestVector(const std::set<float>& x, const std::set<float>& y, const std::set<float>& z)
{
  const std::set<float>* largest = &x;
  int axis = 0;
  if (y.size() > largest->size())
    {
      largest = &y;
      axis = 1;
    }
  else if (z.size() > largest->size())
    {
      largest = &z;
      axis = 2;
    }

  info(format("I believe the axial direction is the {}.", axis));
  return *largest;
}

void
CListModeDataPETSIRD::find_uniqe_values_1D(std::set<float>& values, const std::vector<float>& input)
{
  for (float val : input)
    {
      // std::cout << val << std::endl;
      values.insert(val);
    }
}

void
CListModeDataPETSIRD::find_uniqe_values_2D(std::set<float>& values, const std::vector<std::vector<float>>& input)
{
  for (size_t row = 0; row < input.size(); ++row)
    for (size_t col = 0; col < input[row].size(); ++col)
      values.insert(input[row][col]);
}

int
CListModeDataPETSIRD::figure_out_scanner_blocks_and_rotation_axis(
    std::set<float>& unique_dim1_values,
    std::set<float>& unique_dim2_values,
    std::set<float>& unique_dim3_values,
    const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list)
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

void
CListModeDataPETSIRD::figure_out_block_element_transformations(
    std::set<float>& unique_dim1_values,
    std::set<float>& unique_dim2_values,
    std::set<float>& unique_dim3_values,
    float& radius,
    int& radius_index,
    const int rotation_axis,
    const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list)
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

  for (const auto& module : replicated_module_list)
    {
      for (const auto& el_trans : module.object.detecting_elements.transforms)
        {
          if (radius == 0.0f)
            {
              if (!detect_radius(el_trans))
                {
                  error("Unable to determine radius from translation components.");
                  continue;
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
CListModeDataPETSIRD::figure_out_block_angles(std::set<float>& unique_angle_modules,
                                              const int rot_axis,
                                              const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list)
{
  for (const auto& module : replicated_module_list)
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

bool
CListModeDataPETSIRD::isCylindricalConfiguration(const std::vector<petsird::ReplicatedDetectorModule>& replicated_module_list)
{
  // Determine DOI based on material
  float average_doi = 0.0;
  if (scanner_info->bulk_materials.size() > 0)
    {
      const std::string& material = scanner_info->bulk_materials[0].name;
      if (material.size() > 0)
        average_doi = (material == "BGO") ? 5.0f : (material == "LSO" || material == "LYSO") ? 7.0f : 0.0f;
    }

  const petsird::TypeOfModule type_of_module = replicated_module_list.size();

  if (type_of_module > 1)
    {
      info("Multiple types of PETSIRD modules are not supported. Abord.");
      return false;
    }

  const auto& tof_bin_edges = scanner_info->tof_bin_edges[type_of_module - 1][type_of_module - 1];
  std::cout << "Num. of TOF bins " << tof_bin_edges.NumberOfBins() << std::endl;

  std::set<float> unique_tof_values;
  find_uniqe_values_2D(unique_tof_values, scanner_info->tof_resolution);

  numberOfModules = replicated_module_list[0].NumberOfObjects();
  numberOfElementsIndices = replicated_module_list[0].object.detecting_elements.NumberOfObjects();
  std::set<float> unique_dim1_values, unique_dim2_values, unique_dim3_values;
  std::set<float> unique_tof_resolutions;

  const int rotation_axis = figure_out_scanner_blocks_and_rotation_axis(
      unique_dim1_values, unique_dim2_values, unique_dim3_values, replicated_module_list);
  if (rotation_axis == -1)
    return false;

  const std::set<float>& main_axis = getLargestVector(unique_dim1_values, unique_dim2_values, unique_dim3_values);

  int num_transaxial_blocks = numberOfModules / main_axis.size();
  info(format("I deduce that the scanner has {} transaxial number of blocks", num_transaxial_blocks));

  float radius = 0;
  int radius_indx = -1;

  std::set<float> unique_elements_dim1_values, unique_elements_dim2_values, unique_elements_dim3_values;
  figure_out_block_element_transformations(unique_elements_dim1_values,
                                           unique_elements_dim2_values,
                                           unique_elements_dim3_values,
                                           radius,
                                           radius_indx,
                                           rotation_axis,
                                           replicated_module_list);
  std::set<float> unique_angle_modules;
  figure_out_block_angles(unique_angle_modules, rotation_axis, replicated_module_list);

  std::vector<float> block_angular_spacing;
  if (!get_spacing_uniform(block_angular_spacing, unique_angle_modules, 1e-2)) /// epsilon * 10000) // relax epsilon here
    return false;

  std::vector<float> element_horizontal_spacing, element_vertical_spacing;
  std::set<float> unique_elements_horizontal_values, unique_elements_vertical_values;
  if (radius_indx == 0)
    {
      if (!get_spacing_uniform(element_horizontal_spacing, unique_elements_dim3_values))
        return false;
      if (!get_spacing_uniform(element_vertical_spacing, unique_elements_dim2_values))
        return false;
      unique_elements_horizontal_values = unique_elements_dim3_values;
      unique_elements_vertical_values = unique_elements_dim2_values;
    }
  else
    {
      error("TODO!");
    }

  std::vector<float> block_axial_spacing;
  get_spacing_uniform(block_axial_spacing, main_axis);
  if (block_axial_spacing.size() < 1)
    {
      //      std::set<int>::iterator it = unique_elements_horizontal_values.begin();
      //      std::advance(it,0);
      float begin = *std::next(unique_elements_horizontal_values.begin(), 0);
      //      std::advance(it,unique_elements_horizontal_values.size()-1);
      float end = *std::next(unique_elements_horizontal_values.begin(), unique_elements_horizontal_values.size() - 1);
      block_axial_spacing.push_back(std::abs(end - begin));
    }

  info(format("I counted {} axial blocks with spacing {}", unique_dim3_values.size(), block_axial_spacing[0]));

  // Check if the cyrcle area is less than 5% different from the polygon 
  float expected_circle_area = float(M_PI * radius * radius);;
  float polygon_area = float(0.5f * unique_angle_modules.size() * radius * radius * std::sin(2.f * float(M_PI) / unique_angle_modules.size()));
  info(format("Circle area: {}, Polygon area: {}, pct {}", expected_circle_area, polygon_area, std::abs(expected_circle_area - polygon_area) / expected_circle_area));
  if (std::abs(expected_circle_area - polygon_area) / expected_circle_area < 0.05f)
    {
      info("the cylindical area is more then 95% matching the polygon area. We will predsume a cylindrical configuration.");
      this_scanner_sptr.reset(
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
                  /* doi */ 4, // average_doi,
                  /* ring spacing */
                  element_horizontal_spacing[0], //* 10.f,
                  // bin_size_v
                  element_vertical_spacing[0],// * 10.f,
                  /*intrinsic_tilt_v*/
                  0.f,
                  /*num_axial_blocks_per_bucket_v */
                  unique_dim3_values.size(),
                  /*num_transaxial_blocks_per_bucket_v*/
                  1,
                  /*num_axial_crystals_per_block_v*/
                  unique_elements_horizontal_values.size(),
                  /*num_transaxial_crystals_per_block_v*/
                  unique_elements_vertical_values.size(),
                  /*num_axial_crystals_per_singles_unit_v*/
                  unique_elements_horizontal_values.size(),
                  /*num_transaxial_crystals_per_singles_unit_v*/
                  unique_elements_vertical_values.size(),
                  /*num_detector_layers_v*/
                  1,                                             // num_detector_layers_v
                  scanner_info->energy_resolution_at_511.front(), // energy_resolution_v
                  511,                                           // reference_energy_v
                  tof_bin_edges.NumberOfBins() + 1,
                  (tof_bin_edges.edges[1] - tof_bin_edges.edges[0])*10/2,
                  *unique_tof_values.begin() * 10                                                        // non-TOF
                  ));

                            // 13,
          // 4.056 * 1000 / 13,
          // 555.F); // TODO singles info incorrect
  

  // /* maximum number of timing bins */
  // tof_bin_edges.NumberOfBins(),
  // /* size of basic TOF bin */
  // 10,
  // /* Scanner's timing resolution */
  // *unique_tof_values.begin()));
      return true;
    }
    else
    {
      info("the cylindical area is less then 95% matching the polygon area. We will predsume a non-cylindrical configuration.");
      this_scanner_sptr.reset(
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
                  element_vertical_spacing[0],// * 10.f,
                  /*intrinsic_tilt_v*/
                  0.f,
                  /*num_axial_blocks_per_bucket_v */
                  unique_dim3_values.size(),
                  /*num_transaxial_blocks_per_bucket_v*/
                  1,
                  /*num_axial_crystals_per_block_v*/
                  unique_elements_horizontal_values.size(),
                  /*num_transaxial_crystals_per_block_v*/
                  unique_elements_vertical_values.size(),
                  /*num_axial_crystals_per_singles_unit_v*/
                  unique_elements_horizontal_values.size(),
                  /*num_transaxial_crystals_per_singles_unit_v*/
                  unique_elements_vertical_values.size(),
                  /*num_detector_layers_v*/
                  1,                                             // num_detector_layers_v
                  scanner_info->energy_resolution_at_511.front(), // energy_resolution_v
                  511,                                           // reference_energy_v
                  1,
                  0.F,
                  0.F,                                                                 // non-TOF
                  "BlocksOnCylindrical",                                               // scanner_geometry_v
                  *unique_elements_horizontal_values.begin(),                          // axial_crystal_spacing_v
                  std::round(*unique_elements_vertical_values.begin()), // transaxial_crystal_spacing_v
                  block_axial_spacing.front(),                                         // axial_block_spacing_v
                  radius * block_angular_spacing.front(),                              // transaxial_block_spacing_v
                  ""                                                                   // crystal_map_file_name_v
                  ));
      return false;
    }

  return true;
}

CListModeDataPETSIRD::CListModeDataPETSIRD(const std::string& listmode_filename, bool use_hdf5)
    : use_hdf5(use_hdf5)
{
  CListModeDataBasedOnCoordinateMap::listmode_filename = listmode_filename;

  petsird::Header header;
  if (use_hdf5)
    current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(listmode_filename));
  else
    current_lm_data_ptr.reset(new petsird::binary::PETSIRDReader(listmode_filename));

  current_lm_data_ptr->ReadHeader(header);
  scanner_info = std::make_shared<petsird::ScannerInformation>(header.scanner);
  petsird::ScannerGeometry scanner_geo = scanner_info->scanner_geometry;
  std::vector<petsird::ReplicatedDetectorModule> replicated_module_list = scanner_geo.replicated_modules;

  // Get the first TimeBlock
  // if (
  current_lm_data_ptr->ReadTimeBlocks(curr_time_block);
  // )
  // error("CListModeDataPETSIRD: Could not read the first TimeBlock. Abord.");

  if (std::holds_alternative<petsird::EventTimeBlock>(curr_time_block))
    curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
  else
    error("CListModeDataPETSIRD: holds_alternative not true. Abord.");
  
  if (  isCylindricalConfiguration(replicated_module_list))
    {
      int tof_mash_factor = 1;
      this->set_proj_data_info_sptr(std::const_pointer_cast<const ProjDataInfo>(
          ProjDataInfo::construct_proj_data_info(this_scanner_sptr,
                                                 1,
                                                 this_scanner_sptr->get_num_rings() - 1,
                                                 this_scanner_sptr->get_num_detectors_per_ring() / 2,
                                                 this_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                 /* arc_correction*/ false,
                                                 tof_mash_factor)
              ->create_shared_clone()));
    }
  else
    {
      //      this_scanner_sptr.get_
      DetectorCoordinateMap::det_pos_to_coord_type petsird_map;
      const petsird::TypeOfModule type_of_module = replicated_module_list.size() - 1;
      //      const auto event_energy_bin_edges=scanner_info.event_energy_bin_edges[type_of_module];
      //      const auto num_event_energy_bins = event_energy_bin_edges.NumberOfBins();
      //      error("TODO:GenericScanner");
      for (uint32_t module = 0; module < numberOfModules; module++)
        for (uint32_t elem = 0; elem < numberOfElementsIndices; elem++)
          //            for (uint32_t ener = 0; ener < num_event_energy_bins; ener++) //energy not supported yet
          {
            int index = module * numberOfElementsIndices + elem;
            //              Here we are going to assume that the index = module*numberOfElementsIndices+elem is equal to ax+
            //              num_ax*tang+ num_ax*num_tang*rad then we will pass this to the map sorter (set_detector_map))
            //              therefore we can get ax, tang and rad from index
            /*
             * rad = index/(num_ax*num_tang)
             * tang =(index-rad*num_ax*num_tang)/num_ax -num_tang/2
             * ax = index mod num_ax
             */

            int rad_pos = index / (this_scanner_sptr->get_num_rings() * this_scanner_sptr->get_num_detectors_per_ring());
            int ax_pos = (index - rad_pos * this_scanner_sptr->get_num_rings() * this_scanner_sptr->get_num_detectors_per_ring())
                         / this_scanner_sptr->get_num_detectors_per_ring();
            int tang_pos = index % this_scanner_sptr->get_num_detectors_per_ring();

            DetectionPosition<> detpos(tang_pos, ax_pos, rad_pos);
            petsird::ExpandedDetectionBin expanded_detection_bin{ module, elem, 1 };

            auto box_shape = petsird_helpers::geometry::get_detecting_box(*scanner_info, type_of_module, expanded_detection_bin);
            CartesianCoordinate3D<float> mean_coord;

            for (auto& corner : box_shape.corners)
              { // if STIR (z,y,x) -> PETSIRD (-y, -x, z) pheraps  the order below needs to be changed
                mean_coord.x() = +corner.c[0] / box_shape.corners.size() * 10 ;
                mean_coord.y() = +corner.c[1] / box_shape.corners.size() * 10 ;
                mean_coord.z() = +corner.c[2] / box_shape.corners.size();
              }

            //       save  mean pos into map
            petsird_map[detpos] = mean_coord;
            std::cout << "NIKOSSS " << detpos.radial_coord() << ", " << detpos.axial_coord() << ", " << detpos.tangential_coord() << ", "
                      << mean_coord.x() << ", " << mean_coord.y() << ", " << mean_coord.z() << ", " << std::endl;
          }

      this->map.reset(new DetectorCoordinateMap(petsird_map));

      int tof_mash_factor = 1;
      this->set_proj_data_info_sptr(std::const_pointer_cast<const ProjDataInfo>(
          ProjDataInfo::construct_proj_data_info(this_scanner_sptr,
                                                 1,
                                                 this_scanner_sptr->get_num_rings() - 1,
                                                 this_scanner_sptr->get_num_detectors_per_ring() / 2,
                                                 this_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                 /* arc_correction*/ false,
                                                 tof_mash_factor)
              ->create_shared_clone()));

    }

  shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
  // Only PET scanners supported
  _exam_info_sptr->imaging_modality = ImagingModality::PT;
  _exam_info_sptr->originating_system = std::string("PETSIRD_defined_scanner");
  // _exam_info_sptr->set_low_energy_thres(scanner_i);
  // _exam_info_sptr->set_high_energy_thres(this->root_file_sptr->get_up_energy_thres());

  this->exam_info_sptr = _exam_info_sptr;

  // N.E.: In my experience the first time block is always empty.
  // So I use this unncessessary call to skip to the next.
  if (this->open_lm_file() == Succeeded::no)
    {
      error("CListModeDataPETSIRD: Could not open listmode file " + listmode_filename + "\n");
    }
}

Succeeded
CListModeDataPETSIRD::open_lm_file() const
{
  // current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(listmode_filename));
  if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
    return Succeeded::no;
  curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
  return Succeeded::yes;
}

shared_ptr<CListRecord>
CListModeDataPETSIRD::get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordPETSIRD(scanner_info, this_scanner_sptr));
  std::dynamic_pointer_cast<CListRecordPETSIRD>(sptr)->event().set_scanner_sptr(
      this->get_proj_data_info_sptr()->get_scanner_sptr());
  std::dynamic_pointer_cast<CListRecordPETSIRD>(sptr)->event().set_PETSIRD_ranges(numberOfModules, numberOfElementsIndices);
  // std::dynamic_pointer_cast<CListRecordPETSIRD>(sptr)->event_PETSIRD().set_map_sptr(map);

  return sptr;
}

Succeeded
CListModeDataPETSIRD::get_next_record(CListRecord& record_of_general_type) const
{

  auto& record = dynamic_cast<CListRecordPETSIRD&>(record_of_general_type);
  const auto& prompt_list = curr_event_block.prompt_events.at(0).at(0); // TODO: support mulitple pairs of modules.
  const auto& delayed_list = m_has_delayeds ? curr_event_block.delayed_events.at(0).at(0) : prompt_list;

  const auto& event_list = curr_is_prompt ? prompt_list : delayed_list;

  if (event_list.size() == 0)
    return Succeeded::no;

  if (record.init_from_data(event_list.at(curr_event_in_event_block), curr_is_prompt) == Succeeded::no
      || record_of_general_type.time().set_time_in_millisecs(curr_event_block.time_interval.start) == Succeeded::no)
    {
      return Succeeded::no;
    }

  ++curr_event_in_event_block;

  if (curr_event_in_event_block < event_list.size())
    {
      return Succeeded::yes;
    }

  // - Once we hit the size of the vector
  curr_event_in_event_block = 0;

  if (!m_has_delayeds || curr_is_prompt)
    {
      if (m_has_delayeds)
        {
          curr_is_prompt = false;
        }
      else
        {
          if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
            return Succeeded::no;
          curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
        }
    }
  else
    {
      curr_is_prompt = true;
      if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
        return Succeeded::no;
      curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
    }

  return Succeeded::yes;
}

END_NAMESPACE_STIR
