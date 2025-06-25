/*
  Copyright (C) 2018, 2023-2024, University College London
  Copyright (C) 2019-2023, National Physical Laboratory
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities
  \brief Convert SPECT DICOM projection data to Interfile

  Read files based on https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.4.html,
  https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.4.8.html etc

  \author Benjamin Thomas
  \author Daniel Deidda
  \author Kris Thielemans
*/
#include <iostream>
#include <fstream>
#include <memory>

#include <gdcmReader.h>
#include <gdcmStringFilter.h>

#include "stir/info.h"
#include "stir/error.h"
#include "stir/format.h"
#include "stir/warning.h"
#include "stir/Succeeded.h"

enum class EnergyWindowInfo
{
  LowerThreshold,
  UpperThreshold,
  WindowName
};
enum class RadionuclideInfo
{
  CodeValue,
  CodingSchemDesignator,
  CodeMeaning
};
enum class CalibrationInfo
{
};
stir::Succeeded GetDICOMTagInfo(const gdcm::File& file, const gdcm::Tag tag, std::string& dst, const int sequence_idx = 1);
stir::Succeeded
GetEnergyWindowInfo(const gdcm::File& file, const EnergyWindowInfo request, std::string& dst, const int sequence_idx);
stir::Succeeded
GetRadionuclideInfo(const gdcm::File& file, const RadionuclideInfo request, std::string& dst, const int sequence_idx = 1);

class SPECTDICOMData
{
public:
  SPECTDICOMData(const std::string& DICOM_filename) { dicom_filename = DICOM_filename; };
  stir::Succeeded get_interfile_header(std::string& output_header, const std::string& data_filename, const int dataset_num) const;
  stir::Succeeded get_proj_data(const std::string& output_file) const;
  stir::Succeeded open_dicom_file();
  int num_energy_windows = 1;
  int num_frames = 0;
  int num_of_projections = 0;
  std::vector<std::string> energy_window_name;

private:
  // shared_ptr<stir::ProjDataInfo> proj_data_info_sptr;
  std::string dicom_filename;

  float start_angle = 0.0f;
  float angular_step = 0.0f;
  int actual_frame_duration = 0; // frame duration in msec
  int num_of_rotations = 0;
  std::string direction_of_rotation, isotope_name;
  int extent_of_rotation;
  float calibration_factor;
  std::string rotation_radius;

  std::vector<float> lower_en_window_thres;
  std::vector<float> upper_en_window_thres;

  int num_dimensions;
  std::vector<std::string> matrix_labels;
  std::vector<int> matrix_size;
  std::vector<double> pixel_sizes;
};

stir::Succeeded
GetDICOMTagInfo(const gdcm::File& file, const gdcm::Tag tag, std::string& dst, const int sequence_idx)
{

  // Extracts information for a given DICOM tag from a gdcm dataset.
  // Tag contents are returned as a string in dst variable.

  // Tries to read the element associated with the tag. If the read fails, the
  // DataElement should have a ByteValue of NULL.

  try
    {
      const gdcm::DataSet& ds = file.GetDataSet();

      gdcm::StringFilter sf;
      sf.SetFile(file);

      std::stringstream inStream;
      inStream.exceptions(std::ios::badbit);

      // First try and see if this is a standard tag.
      gdcm::DataElement element = ds.GetDataElement(tag);

      if (element.GetByteValue() != NULL)
        {
          dst = sf.ToString(tag);
          return stir::Succeeded::yes;
        }

      // Try: RotationInformationSequence     (0054,0052)
      //      DetectorInformationSequence     (0054,0022)
      //      EnergyWindowInformationSequence (0054,0012)
      std::vector<gdcm::Tag> seqs = { gdcm::Tag(0x0054, 0x0052), gdcm::Tag(0x0054, 0x0022), gdcm::Tag(0x0054, 0x0012) };

      for (const auto& t : seqs)
        {
          try
            {
              const gdcm::DataElement& de = file.GetDataSet().GetDataElement(t);
              const gdcm::SequenceOfItems* sqi = de.GetValueAsSQ();
              if (!sqi)
                {
                  // try next sequence
                  continue;
                }
              const auto sqi_length = sqi->GetNumberOfItems();
              if (static_cast<unsigned>(sequence_idx) > sqi_length)
                {
                  // stir::warning("sequence_id larger than length");
                  // try next sequence
                  continue;
                }
              const gdcm::Item& item = sqi->GetItem(sequence_idx);

              element = item.GetDataElement(tag);

              if (element.GetByteValue() != NULL)
                {
                  dst = sf.ToString(element);
                  return stir::Succeeded::yes;
                }
            }
          catch (...)
            {
              // ignore error and try next sequence
            }
        }
    }
  catch (...)
    {
      stir::error(stir::format(
          "GetDICOMTagInfo: cannot read tag {} sequence index {}",
          [&]() { // using lambda function to leverage the operator<< to transform tag to string
            std::stringstream ss;
            ss << tag;
            return ss.str();
          }(),
          sequence_idx));
      return stir::Succeeded::no;
    }

  return stir::Succeeded::no;
}

stir::Succeeded
GetEnergyWindowInfo(const gdcm::File& file, const EnergyWindowInfo request, std::string& dst, const int sequence_idx)
{

  if (request == EnergyWindowInfo::WindowName)
    {
      return GetDICOMTagInfo(file, gdcm::Tag(0x0054, 0x0018), dst, sequence_idx);
    }

  try
    {
      const gdcm::Tag energy_window_info_seq = gdcm::Tag(0x0054, 0x0012);
      const gdcm::Tag energy_window_range_seq = gdcm::Tag(0x0054, 0x0013);

      const gdcm::Tag lower_energy_window_tag = gdcm::Tag(0x0054, 0x0014);
      const gdcm::Tag upper_energy_window_tag = gdcm::Tag(0x0054, 0x0015);

      // Get Energy Window Info Sequence
      const gdcm::DataElement& de = file.GetDataSet().GetDataElement(energy_window_info_seq);
      const gdcm::SequenceOfItems* sqi = de.GetValueAsSQ();
      const gdcm::Item& item = sqi->GetItem(sequence_idx);

      // Get Energy Window Range Sequence
      const gdcm::DataElement& element = item.GetDataElement(energy_window_range_seq);
      const gdcm::SequenceOfItems* sqi2 = element.GetValueAsSQ();
      if (sqi2->GetNumberOfItems() > 1)
        stir::warning("Energy window sequence contains more than 1 window. Ignoring all later ones");
      const gdcm::Item& item2 = sqi2->GetItem(1);

      // std::cout << item2 << std::endl;

      gdcm::DataElement window_element;

      if (request == EnergyWindowInfo::LowerThreshold)
        window_element = item2.GetDataElement(lower_energy_window_tag);
      else
        window_element = item2.GetDataElement(upper_energy_window_tag);

      if (window_element.GetByteValue() != NULL)
        {
          gdcm::StringFilter sf;
          sf.SetFile(file);
          dst = sf.ToString(window_element);
          return stir::Succeeded::yes;
        }
    }
  catch (...)
    {
      stir::warning("GetEnergyWindowInfo: cannot read energy info");
      return stir::Succeeded::no;
    }

  return stir::Succeeded::no;
}

stir::Succeeded
GetRadionuclideInfo(const gdcm::File& file, const RadionuclideInfo request, std::string& dst, const int sequence_idx)
{

  //  if (request == RadionuclideInfo::CodeMeaning){
  //    return GetDICOMTagInfo(file, gdcm::Tag(0x0008,0x0018), dst);
  //  }

  try
    {
      const gdcm::Tag radiopharm_info_seq_tag = gdcm::Tag(0x0054, 0x0016);
      const gdcm::Tag radionuclide_code_seq_tag = gdcm::Tag(0x0054, 0x0300);

      const gdcm::Tag radionuclide_codemeaning_tag = gdcm::Tag(0x0008, 0x0104);
      //    const gdcm::Tag upper_radionuclide_tag = gdcm::Tag(0x0054,0x0015);

      // Get Radiopharmaceutical Info Sequence
      const gdcm::DataElement& de = file.GetDataSet().GetDataElement(radiopharm_info_seq_tag);
      const gdcm::SequenceOfItems* sqi = de.GetValueAsSQ();
      const gdcm::Item& item = sqi->GetItem(sequence_idx);

      // Get Radiopnuclide Code Sequence
      gdcm::DataElement nuclide_element;
      const gdcm::DataElement& element = item.GetDataElement(radionuclide_code_seq_tag);
      if (element.GetVL() > 0)
        {
          const gdcm::SequenceOfItems* sqi2 = element.GetValueAsSQ();
          const gdcm::Item& item2 = sqi2->GetItem(sequence_idx);
          // std::cout<< "num items"<< sqi2->GetNumberOfItems();
          // std::cout << item2 << std::endl;

          //    gdcm::DataElement nuclide_element;
          nuclide_element = item2.GetDataElement(radionuclide_codemeaning_tag);
        }

      if (nuclide_element.GetByteValue() != NULL)
        {
          gdcm::StringFilter sf;
          sf.SetFile(file);
          dst = sf.ToString(nuclide_element);
          return stir::Succeeded::yes;
        }
    }
  catch (...)
    {
      stir::warning("GetRadionuclideInfo: cannot read radiopharmaceutical info");
      dst = "";
      return stir::Succeeded::no;
    }

  return stir::Succeeded::no;
}

stir::Succeeded
SPECTDICOMData::open_dicom_file()
{

  stir::info(stir::format("SPECTDICOMData: opening file {}", dicom_filename));

  std::unique_ptr<gdcm::Reader> DICOM_reader(new gdcm::Reader);
  DICOM_reader->SetFileName(dicom_filename.c_str());

  try
    {
      if (!DICOM_reader->Read())
        {
          stir::error(stir::format("SPECTDICOMData: cannot read file {}", dicom_filename));
          // return stir::Succeeded::no;
        }
    }
  catch (const std::string& e)
    {
      std::cerr << e << std::endl;
      return stir::Succeeded::no;
    }

  const gdcm::File& file = DICOM_reader->GetFile();

  std::cout << std::endl;

  std::string patient_name;
  if (GetDICOMTagInfo(file, gdcm::Tag(0x0010, 0x0010), patient_name) == stir::Succeeded::yes)
    {
      std::cout << "Patient name: " << patient_name << std::endl;
    }
  std::string no_of_proj_as_str;
  std::string start_angle_as_string;
  std::string angular_step_as_string;
  std::string extent_of_rotation_as_string;
  std::string radius_as_string;
  std::string actual_frame_duration_as_string;
  std::string calib_factor_as_string;

  std::string matrix_size_as_string;
  std::string pixel_size_as_string;
  std::string lower_window_as_string;
  std::string upper_window_as_string;

  {
    std::string str;
    if (GetDICOMTagInfo(file, gdcm::Tag(0x0028, 0x0008), str) == stir::Succeeded::yes)
      {
        num_frames = std::stoi(str);
        std::cout << "Number of frames: " << num_frames << std::endl;
      }
  }

  this->num_of_projections = 1;
  this->calibration_factor = -1;
  if (GetRadionuclideInfo(file, RadionuclideInfo::CodeMeaning, isotope_name) == stir::Succeeded::yes)
    {
      std::cout << "Isotope name: " << isotope_name << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0018, 0x1242), actual_frame_duration_as_string) == stir::Succeeded::yes)
    {
      actual_frame_duration = std::stoi(actual_frame_duration_as_string);
    }

  {
    this->num_energy_windows = 0;
    std::string str;
    if (GetDICOMTagInfo(file, gdcm::Tag(0x0054, 0x0011), str) == stir::Succeeded::yes)
      this->num_energy_windows = std::stoi(str);
  }
  if (this->num_energy_windows == 0)
    {
      std::cout << "No energy window information found\n";
    }
  else
    {
      energy_window_name.resize(num_energy_windows);
      lower_en_window_thres.resize(num_energy_windows);
      upper_en_window_thres.resize(num_energy_windows);
      for (int w = 1; w <= num_energy_windows; ++w)
        {
          if (GetEnergyWindowInfo(file, EnergyWindowInfo::WindowName, energy_window_name[w - 1], w) == stir::Succeeded::yes)
            {
              std::cout << "Energy window: " << energy_window_name[w - 1] << std::endl;
            }
          else
            energy_window_name[w - 1] = "en" + std::to_string(w);

          if (GetEnergyWindowInfo(file, EnergyWindowInfo::LowerThreshold, lower_window_as_string, w) == stir::Succeeded::yes)
            {
              lower_en_window_thres[w - 1] = std::stof(lower_window_as_string);
              std::cout << "Lower energy window limit: " << std::fixed << std::setprecision(6) << lower_en_window_thres[w - 1]
                        << std::endl;
            }

          if (GetEnergyWindowInfo(file, EnergyWindowInfo::UpperThreshold, upper_window_as_string, w) == stir::Succeeded::yes)
            {
              upper_en_window_thres[w - 1] = std::stof(upper_window_as_string);
              std::cout << "Upper energy window limit: " << std::fixed << std::setprecision(6) << upper_en_window_thres[w - 1]
                        << std::endl;
            }
        }
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0054, 0x0053), no_of_proj_as_str) == stir::Succeeded::yes)
    {
      num_of_projections = std::stoi(no_of_proj_as_str);
      std::cout << "Number of projections: " << num_of_projections << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0018, 0x1140), direction_of_rotation) == stir::Succeeded::yes)
    {
      if (direction_of_rotation == "CC")
        direction_of_rotation = "CCW";

      std::cout << "Direction of rotation: " << direction_of_rotation << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0054, 0x0200), start_angle_as_string) == stir::Succeeded::yes)
    {
      start_angle = std::stof(start_angle_as_string);
      std::cout << "Starting angle: " << std::fixed << std::setprecision(6) << start_angle << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0054, 0x1322), calib_factor_as_string) == stir::Succeeded::yes)
    {
      calibration_factor = std::stof(calib_factor_as_string);
      std::cout << "calibration factor: " << std::fixed << std::setprecision(6) << calibration_factor << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0018, 0x1144), angular_step_as_string) == stir::Succeeded::yes)
    {
      angular_step = std::stof(angular_step_as_string);
      std::cout << "Angular step: " << std::fixed << std::setprecision(6) << angular_step << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0018, 0x1143), extent_of_rotation_as_string) == stir::Succeeded::yes)
    {
      extent_of_rotation = std::stoi(extent_of_rotation_as_string);
      std::cout << "Rotation extent: " << extent_of_rotation << std::endl;
    }
  else
    {
      extent_of_rotation = 360;
      stir::warning("Rotation was not present in DICOM file. Setting it to 360");
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0018, 0x1142), radius_as_string) == stir::Succeeded::yes)
    {
      rotation_radius = (radius_as_string);
      char slash = '\\';
      char comma = ',';
      std::cout << "Radius: " << radius_as_string << " " << slash << std::endl;
      std::replace(rotation_radius.begin(), rotation_radius.end(), slash, comma);
    }

  num_dimensions = 2;

  matrix_labels.push_back("axial coordinate");
  matrix_labels.push_back("bin coordinate");

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0028, 0x0010), matrix_size_as_string) == stir::Succeeded::yes)
    {
      matrix_size.push_back(std::stoi(matrix_size_as_string));
      std::cout << "Matrix size [1]: " << matrix_size_as_string << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0028, 0x0011), matrix_size_as_string) == stir::Succeeded::yes)
    {
      matrix_size.push_back(std::stoi(matrix_size_as_string));
      std::cout << "Matrix size [2]: " << matrix_size_as_string << std::endl;
    }

  if (GetDICOMTagInfo(file, gdcm::Tag(0x0028, 0x0030), pixel_size_as_string) == stir::Succeeded::yes)
    {
      std::cout << "Pixel size: " << pixel_size_as_string << std::endl;
      size_t curr = 0, prev = 0;

      while (curr != std::string::npos)
        {
          curr = pixel_size_as_string.find('\\', prev);
          std::string found = pixel_size_as_string.substr(prev, curr - prev);
          //    std::cout << "found = " << found << std::endl;
          pixel_sizes.push_back(std::stof(found));
          prev = curr + 1;
        }
    }

  std::cout << std::endl;

  return stir::Succeeded::yes;
}

stir::Succeeded
SPECTDICOMData::get_interfile_header(std::string& output_header, const std::string& data_filename, const int dataset_num) const
{

  std::string data_filename_only = data_filename;

  const size_t last_slash_idx = data_filename_only.find_last_of("\\/");
  if (std::string::npos != last_slash_idx)
    {
      data_filename_only.erase(0, last_slash_idx + 1);
    }

  std::stringstream ss;

  ss << "!INTERFILE  :=" << std::endl;
  ss << "!imaging modality := nucmed" << std::endl;
  ss << "!version of keys := 3.3" << std::endl;
  ss << "name of data file := " << data_filename_only << std::endl;
  ss << "data offset in bytes := "
     << this->matrix_size.at(0) * this->matrix_size.at(1) * this->num_of_projections * dataset_num * 4 // float hard-wired
     << std::endl;
  ss << std::endl;

  ss << "!GENERAL IMAGE DATA :=" << std::endl;
  // We will write planar data as SPECT (i.e. "tomographic") anyway
  ss << "!type of data := Tomographic" << std::endl;
  ss << "imagedata byte order := LITTLEENDIAN" << std::endl;
  ss << "!number format := float" << std::endl;
  ss << "!number of bytes per pixel := 4" << std::endl;
  ss << "calibration factor:= " << this->calibration_factor << std::endl;
  if (!this->isotope_name.empty())
    ss << "isotope name:= " << this->isotope_name << std::endl;
  ss << std::endl;

  // one energy window per header
  ss << "number of energy windows :=1\n";
  ss << "energy window lower level[1] := " << this->lower_en_window_thres[dataset_num] << "\n";
  ss << "energy window upper level[1] := " << this->upper_en_window_thres[dataset_num] << "\n";
  ss << "!SPECT STUDY (General) :=" << std::endl;
  ss << "number of dimensions := 2" << std::endl;
  ss << "matrix axis label [2] := axial coordinate" << std::endl;
  ss << "!matrix size [2] := " << this->matrix_size.at(1) << std::endl;
  ss << "!scaling factor (mm/pixel) [2] := " << this->pixel_sizes.at(1) << std::endl;
  ss << "matrix axis label [1] := bin coordinate" << std::endl;
  ss << "!matrix size [1] := " << this->matrix_size.at(0) << std::endl;
  ss << "!scaling factor (mm/pixel) [1] := " << this->pixel_sizes.at(0) << std::endl;
  ss << "!number of projections := " << this->num_of_projections << std::endl;
  ss << "number of time frames := 1" << std::endl;
  ss << "!image duration (sec)[1] := " << this->num_of_projections * this->actual_frame_duration / 1000 << std::endl;
  ss << "!extent of rotation := " << this->extent_of_rotation << std::endl;
  ss << "!process status := acquired" << std::endl;
  ss << std::endl;

  ss << "!SPECT STUDY (acquired data) :=" << std::endl;
  if (!this->direction_of_rotation.empty())
    ss << "!direction of rotation := " << this->direction_of_rotation << std::endl;
  ss << "start angle := " << this->start_angle << std::endl;

  {
    if (this->rotation_radius.find(",") != std::string::npos)
      {
        ss << "orbit := non-circular" << std::endl;
        ss << "radii := {" << this->rotation_radius << "}" << std::endl;
        std::cout << "orbit := non-circular" << std::endl;
      }
    else
      {
        ss << "orbit := circular" << std::endl;
        ss << "radius := " << this->rotation_radius << "" << std::endl;
      }
  }
  ss << std::endl;

  ss << "!END OF INTERFILE :=";

  output_header = ss.str();

  return stir::Succeeded::yes;
}

stir::Succeeded
SPECTDICOMData::get_proj_data(const std::string& output_file) const
{

  std::unique_ptr<gdcm::Reader> DICOM_reader(new gdcm::Reader);
  DICOM_reader->SetFileName(dicom_filename.c_str());

  try
    {
      if (!DICOM_reader->Read())
        {
          stir::error(stir::format("SPECTDICOMData: cannot read file {}", dicom_filename));
          // return stir::Succeeded::no;
        }
    }
  catch (const std::string& e)
    {
      std::cerr << e << std::endl;
      return stir::Succeeded::no;
    }

  const gdcm::File& file = DICOM_reader->GetFile();

  const gdcm::DataElement& de = file.GetDataSet().GetDataElement(gdcm::Tag(0x7fe0, 0x0010));
  const gdcm::ByteValue* bv = de.GetByteValue();
  if (!bv)
    {
      stir::warning("GDCM GetByteValue failed on x7fe0, 0x0010");
      return stir::Succeeded::no;
    }
  /*
  std::string tmpFile = "tmp.s";
  std::ofstream outfile(tmpFile.c_str(), std::ios::out | std::ios::binary);

  if (!outfile.is_open()) {
    std::cerr << "Unable to write proj data to " << tmpFile;
    return stir::Succeeded::no;
  }
  bv->WriteBuffer(outfile);
  outfile.close();*/

  uint64_t len0 = (uint64_t)bv->GetLength() / 2;
  std::cout << "Number of data points = " << len0 << std::endl;

  std::vector<float> pixel_data_as_float;

  uint16_t* ptr = (uint16_t*)bv->GetPointer();

  uint64_t ct = 0;
  while (ct < len0)
    {
      uint16_t val = *ptr;
      pixel_data_as_float.push_back((float)val);
      ptr++;
      ct++;
    }

  std::ofstream final_out(output_file.c_str(), std::ios::out | std::ofstream::binary);
  final_out.write(reinterpret_cast<char*>(&pixel_data_as_float[0]), pixel_data_as_float.size() * sizeof(float));
  final_out.close();

  return stir::Succeeded::yes;
}

// remove spaces and tabs etc
static std::string
convert_to_filename(const std::string& s)
{
  char const* const white_space = " \t_";
  // skip white space
  const auto cp = s.find_first_not_of(white_space, 0);
  if (cp == std::string::npos)
    return std::string();
  // remove trailing white spaces
  const auto eos = s.find_last_not_of(white_space);
  auto trimmed = s.substr(cp, eos - cp + 1);
  // replace remaining spaces with '_'
  std::replace(trimmed.begin(), trimmed.end(), ' ', '_');
  return trimmed;
}

int
main(int argc, char* argv[])
{

  if (argc != 3 && argc != 4)
    {
      std::cerr << "Usage: " << argv[0] << " <output_interfile_prefix> sinogram(dcm)> [is_planar?]\n"
                << "Note: the is_planar value is ignored. All data will be written as SPECT.\n";
      exit(EXIT_FAILURE);
    }

  SPECTDICOMData spect(argv[2]);
  const std::string output_prefix(argv[1]);

  try
    {
      if (spect.open_dicom_file() == stir::Succeeded::no)
        {
          std::cerr << "Failed to read!" << std::endl;
          return EXIT_FAILURE;
        }
    }
  catch (const std::exception& e)
    {
      std::cerr << e.what() << std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Read successfully!" << std::endl;

  if (spect.num_of_projections * spect.num_energy_windows != spect.num_frames)
    stir::warning(stir::format("num_projections ({}) * num_energy_windows ({}) != num_frames ({})\n"
                               "This could be gated or dynamic data. The interfile header(s) will be incorrect!",
                               spect.num_of_projections,
                               spect.num_energy_windows,
                               spect.num_frames));
  const std::string data_filename(output_prefix + ".s");
  stir::Succeeded s = spect.get_proj_data(data_filename);

  for (int w = 0; w < spect.num_energy_windows; ++w)
    {
      std::string header;
      s = spect.get_interfile_header(header, data_filename, w);

      const std::string header_filename
          = output_prefix + "_en_" + std::to_string(w + 1) + "_" + convert_to_filename(spect.energy_window_name[w]) + ".hdr";
      stir::info("Writing '" + header_filename + "'");
      std::filebuf fb;
      fb.open(header_filename, std::ios::out);
      std::ostream os(&fb);
      os << header;
      fb.close();
    }

  return EXIT_SUCCESS;
}
