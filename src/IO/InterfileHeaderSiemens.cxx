/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-04-30, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, 2016 University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup InterfileIO 
  \brief implementations for the stir::InterfileHeaderSiemens class

  \author Kris Thielemans
  \author PARAPET project
*/

#include "stir/IO/InterfileHeaderSiemens.h"
#include "stir/ExamInfo.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/PatientPosition.h"
#include "stir/ImagingModality.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <numeric>
#include <functional>

#ifndef STIR_NO_NAMESPACES
using std::binary_function;
using std::pair;
using std::sort;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
#endif

START_NAMESPACE_STIR

InterfileHeaderSiemens::InterfileHeaderSiemens()
     : InterfileHeader()
{
  // always PET
  exam_info_sptr->imaging_modality = ImagingModality::PT;
  
  byte_order_values.push_back("LITTLEENDIAN");
  byte_order_values.push_back("BIGENDIAN");
  
  PET_data_type_values.push_back("Emission");
  PET_data_type_values.push_back("Transmission");
  PET_data_type_values.push_back("Blank");
  PET_data_type_values.push_back("AttenuationCorrection");
  PET_data_type_values.push_back("Normalisation");
  PET_data_type_values.push_back("Image");
  
  for (int patient_position_idx = 0; patient_position_idx <= PatientPosition::unknown_position; ++patient_position_idx)
    {
      PatientPosition pos((PatientPosition::PositionValue)patient_position_idx);
      patient_position_values.push_back(pos.get_position_as_string());
    }
  patient_position_index = static_cast<int>(patient_position_values.size()); //unknown
  byte_order_index = 1;//  file_byte_order = ByteOrder::big_endian;

  // need to default to PET for backwards compatibility
  this->exam_info_sptr->imaging_modality = ImagingModality::PT;
  //type_of_data_index = 6; // PET
  PET_data_type_index = 5; // Image

  num_dimensions = 2; // set to 2 to be compatible with Interfile version 3.3 (which doesn't have this keyword)
  matrix_labels.resize(num_dimensions);
  matrix_size.resize(num_dimensions);
  pixel_sizes.resize(num_dimensions, 1.);

  data_offset_each_dataset.resize(1, 0UL);

  data_offset = 0UL;


  /*add_key("type of data", 
          KeyArgument::ASCIIlist,
          (KeywordProcessor)&InterfileHeaderSiemens::set_type_of_data,
          &type_of_data_index, 
          &type_of_data_values);*/

  add_key("%patient orientation",
	  KeyArgument::ASCIIlist,
	  &patient_position_index,
	  &patient_position_values);
  
  add_key("image data byte order", 
    KeyArgument::ASCIIlist,
    &byte_order_index, 
    &byte_order_values);
  
  add_key("scale factor (mm/pixel)", 
    KeyArgument::DOUBLE, &pixel_sizes);

  // only a single time frame supported by Siemens currently
  num_time_frames = 1;
  image_relative_start_times.resize(1, 0.);
  // already added in InterfileHeader but need to override as Siemens doesn't use the "vectored" keys
  // i.e. image duration (sec)[1]
  add_key("image relative start time (sec)", &image_relative_start_times[0]);
  image_durations.resize(1, 0.);
  add_key("image duration (sec)", &image_durations[0]);
}

bool InterfileHeaderSiemens::post_processing()
{

  if (InterfileHeader::post_processing() == true)
    return true;

  /*if(type_of_data_index<0)
    {
      warning("Interfile Warning: 'type_of_data' keyword required");
      return true;
    }*/

  if (patient_position_index<0 )
    return true;

  // note: has to be done after InterfileHeader::post_processing as that sets it as well
  exam_info_sptr->patient_position = PatientPosition((PatientPosition::PositionValue)patient_position_index);

  file_byte_order = byte_order_index==0 ? 
    ByteOrder::little_endian : ByteOrder::big_endian;

  return false;
}

void InterfileHeaderSiemens::set_type_of_data()
{
  set_variable();
#if 0
  // already done below
    {
      add_key("PET data type", 
              KeyArgument::ASCIIlist,
              &PET_data_type_index, 
              &PET_data_type_values);
      add_key("process status", 
              KeyArgument::NONE,	&KeyParser::do_nothing);
      add_key("IMAGE DATA DESCRIPTION", 
              KeyArgument::NONE,	&KeyParser::do_nothing);
      // TODO rename keyword 
      add_key("data offset in bytes", 
	      KeyArgument::ULONG,	&data_offset_each_dataset);

    }
#endif
}


/**********************************************************************/

InterfileRawDataHeaderSiemens::InterfileRawDataHeaderSiemens()
     : InterfileHeaderSiemens()
{
  // first set to some crazy values
  num_segments = -1;
  num_rings = -1;
  maximum_ring_difference = -1;
  axial_compression = -1;
  add_key("number of rings", &num_rings);

  add_key("%axial compression", &axial_compression);
  add_key("%maximum ring difference", &maximum_ring_difference);  
  add_key("%number of segments", &num_segments);
  add_key("%segment table", KeyArgument::LIST_OF_INTS, &segment_table);
  add_key("%number of tof time bins", &num_tof_bins);
  
  num_energy_windows = -1;
  add_key("number of energy windows",
    KeyArgument::INT, (KeywordProcessor)&InterfileRawDataHeaderSiemens::read_num_energy_windows, &num_energy_windows);
  add_key("%energy window lower level (keV)",
    KeyArgument::FLOAT, &lower_en_window_thresholds);

  add_key("%energy window upper level (keV)",
    KeyArgument::FLOAT, &upper_en_window_thresholds);

  add_key("PET data type",
	  KeyArgument::ASCIIlist,
	  &PET_data_type_index,
	  &PET_data_type_values);

  // TODO should add data format:=CoincidenceList|sinogram and then check its value
  add_key("process status",
	  KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("IMAGE DATA DESCRIPTION",
	  KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("data offset in bytes",
	  KeyArgument::ULONG, &data_offset_each_dataset);

  add_key("%comment", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%sms-mi header name space", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%listmode header file", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%listmode data file", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%compressor version", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%study date (yyyy", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%study time (hh", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("isotope name", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("isotope gamma halflife (sec)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("isotope branching factor", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("radiopharmaceutical", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%tracer injection date (yyyy", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%tracer injection time (hh", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("relative time of tracer injection (sec)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("tracer activity at time of injection (bq)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("injected volume (ml)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("horizontal bed translation", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("start horizontal bed position (mm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("end horizontal bed position (mm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("start vertical bed position (mm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%coincidence window width (ns)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("gantry tilt angle (degrees)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("method of attenuation correction", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("method of scatter correction", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%method of random correction", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%decay correction", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%decay correction reference date (yyyy", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%decay correction reference time (hh", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("decay correction factor", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("scatter fraction (%)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("scan data type description", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("total prompts events", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("total prompts", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%total randoms", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%total net trues", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%image duration from timing tags (msec)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%gim loss fraction", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%pdr loss fraction", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%detector block singles", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%total uncorrected singles rate", KeyArgument::NONE, &KeyParser::do_nothing);

}


void InterfileRawDataHeaderSiemens::read_num_energy_windows()
{
  set_variable();

  lower_en_window_thresholds.resize(num_energy_windows, -1.);
  upper_en_window_thresholds.resize(num_energy_windows, -1.);
}

bool InterfileRawDataHeaderSiemens::post_processing()
{
  if (InterfileHeaderSiemens::post_processing() == true)
    return true;

  if (standardise_interfile_keyword(PET_data_type_values[PET_data_type_index]) != "emission")
  { warning("Interfile error: expecting emission data\n");  return true; }

  if (num_energy_windows >= 1)
    {
    // TODO support more energy windows
    exam_info_sptr->set_high_energy_thres(upper_en_window_thresholds[0]);
    exam_info_sptr->set_low_energy_thres(lower_en_window_thresholds[0]);
    }

  // handle scanner

  shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name(get_exam_info_ptr()->originating_system));
  if (scanner_sptr->get_type() == Scanner::Unknown_scanner)
    {
      error("scanner not recognised from originating system");
    }
  // consistency check with values of the scanner
  if (num_rings != scanner_sptr->get_num_rings())
    {
      error("Interfile warning: 'number of rings' (%d) is expected to be %d.\n",
            num_rings, scanner_sptr->get_num_rings());
    }

  data_info_ptr =
    ProjDataInfo::construct_proj_data_info(scanner_sptr,
      axial_compression, maximum_ring_difference,
      num_views, num_bins,
      is_arccorrected);

  // handle segments
  {
    if (num_segments != segment_table.size())
      {
        error("Interfile warning: 'number of segments' and length of 'segment table' are not consistent");
      }
    // same order as in stir_ecat7
    // Siemens always stores segments as 0, -1, +1, ...
    segment_sequence.resize(num_segments);
    segment_sequence[0] = 0;
    for (int segment_num = 1; segment_num <= num_segments/2; ++segment_num)
      {
        segment_sequence[2 * segment_num - 1] = -segment_num;
        segment_sequence[2 * segment_num] = segment_num;
      }
    //XXX check if order here and segment_table are consistent
  }

  return false;
}



/**********************************************************************/

InterfilePDFSHeaderSiemens::InterfilePDFSHeaderSiemens()
  : InterfileRawDataHeaderSiemens()
{
  add_key("number of scan data types",
    KeyArgument::INT, (KeywordProcessor)&InterfilePDFSHeaderSiemens::read_scan_data_types, &num_scan_data_types);
  // scan data type size depends on the previous field
  // scan data type description[1]: = prompts
  // scan data type description[2] : = randoms
  add_key("scan data type description", KeyArgument::ASCII, &scan_data_types);

  // scan data type size depends on the previous field
  // data offset in bytes[1] : = 24504
  //  data offset in bytes[2] : = 73129037

  add_key("%total number of sinograms", &total_num_sinograms);
  add_key("%compression", &compression_as_string);
  add_key("applied corrections",
    KeyArgument::LIST_OF_ASCII, &applied_corrections);

  add_key("%number of buckets",
    KeyArgument::INT, (KeywordProcessor)&InterfilePDFSHeaderSiemens::read_bucket_singles_rates, &num_buckets);
  add_key("%bucket singles rate", KeyArgument::INT, &bucket_singles_rates);
  
  
}


void InterfilePDFSHeaderSiemens::read_scan_data_types()
{
  set_variable();

  scan_data_types.resize(num_scan_data_types);
  data_offset_each_dataset.resize(num_scan_data_types);

}

void InterfilePDFSHeaderSiemens::read_bucket_singles_rates()
{
  set_variable();

  bucket_singles_rates.resize(num_buckets);
}

int InterfilePDFSHeaderSiemens::find_storage_order()
{

  if (num_dimensions != 3)
    {
    warning("Interfile error: expecting 3D data ");
    stop_parsing();
    return true;
    }

  if ((matrix_size[0].size() != 1) ||
    (matrix_size[1].size() != 1) ||
    (matrix_size[2].size() != 1))
    {
    error("Interfile error: strange values for the matrix_size keyword(s)");
    }
  if (matrix_labels[0] != "bin")
    {
    // use error message with index [1] as that is what the user sees.
    warning("Interfile error: expecting 'matrix axis label[1] := bin'\n");
    stop_parsing();
    return true;
    }
  num_bins = matrix_size[0][0];

  if (matrix_labels[1] == "projection" && matrix_labels[2] == "plane")
    {
    storage_order = ProjDataFromStream::Segment_AxialPos_View_TangPos;
    num_views = matrix_size[1][0];
    }
  else
    {
    warning("Interfile error: matrix labels not in expected (or supported) format\n");
    stop_parsing();
    return true;
    }

  return false;

}


bool InterfilePDFSHeaderSiemens::post_processing()
{
  // check for arc-correction
  if (applied_corrections.size() == 0)
    {
    warning("\nParsing Interfile header for projection data: \n"
      "\t'applied corrections' keyword not found. Assuming non-arc-corrected data\n");
    is_arccorrected = false;
    }
  else
    {
    is_arccorrected = false;
    for (
      std::vector<string>::const_iterator iter = applied_corrections.begin();
      iter != applied_corrections.end();
      ++iter)
      {
        const string correction = standardise_keyword(*iter);
        if (correction == "arc correction" || correction == "arc corrected")
          {
            is_arccorrected = true;
            break;
          }
        else if (correction != "none")
          warning("\nParsing Interfile header for projection data: \n"
            "\t value '%s' for keyword 'applied corrections' ignored\n",
            correction.c_str());
      }
    }

  if (find_storage_order())
    {
      error("Interfile error determining storage order");
    }

  // can only do this now after the previous things were set
  if (InterfileRawDataHeaderSiemens::post_processing() == true)
    return true;

  compression = (standardise_interfile_keyword(compression_as_string) == "on");

  return false;
}

/**********************************************************************/

InterfileListmodeHeaderSiemens::InterfileListmodeHeaderSiemens()
  : InterfileRawDataHeaderSiemens()
{
  // need to set this to construct the correct proj_data_info
  is_arccorrected = false;
  // need to set this for InterfileHeader::post_processing()
  // but will otherwise be ignored
  bytes_per_pixel = 4;
  for (unsigned int dim = 0; dim != matrix_size.size(); ++dim)
    {
      matrix_size[dim].resize(1, 1);
    }
/*
 keywords different from a sinogram header (in alphabetical order)
< %LM event and tag words format (bits):=32
< %SMS-MI header name space:=PETLINK bin address

< %number of projections:=344
< %number of views:=252
< %preset type:=time
< %preset unit:=seconds
< %preset value:=900
< %singles polling interval (sec):=2
< %singles polling method:=instantaneous
< %singles scale factor:=8
< %time_sync:=25934299
< %timing tagwords interval (msec):=1
< %total listmode word counts:=331257106
< %total number of singles blocks:=224
< PET scanner type:=cylindrical
< bin size (cm):=0.20445

< data format:=CoincidenceList

< distance between rings (cm):=0.40625
< end horizontal bed position (mm):=0
< gantry crystal radius (cm):=32.8
< gantry tilt angle (degrees):=0

> gantry tilt angle (degrees):=0.0
< septa state:=none
< transaxial FOV diameter (cm):=59.6
*/

  add_key("%number of projections", &num_bins);
  add_key("%number of views", &num_views);
  // override as listmode uses the non-vectored key
  data_offset_each_dataset.resize(1, 0UL);
  add_key("data offset in bytes", &data_offset_each_dataset[0]);

  add_key("%bed zero offset (mm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("pet scanner type", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("transaxial fov diameter (cm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("distance between rings (cm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("gantry crystal radius (cm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("bin size (cm)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("septa state", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%tof mashing factor", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%preset type", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%preset value", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%preset unit", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%total listmode word counts", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%coincidence list data", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%lm event and tag words format (bits)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%timing tagwords interval (msec)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%singles polling method", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%singles polling interval (sec)", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%singles scale factor", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%total number of singles blocks", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%time sync", KeyArgument::NONE, &KeyParser::do_nothing);
  add_key("%comment", KeyArgument::NONE, &KeyParser::do_nothing);
  }

int InterfileListmodeHeaderSiemens::find_storage_order()
{
  // always...
  storage_order = ProjDataFromStream::Segment_AxialPos_View_TangPos;
    
  return false;
}



bool InterfileListmodeHeaderSiemens::post_processing()
{
  if (find_storage_order())
    {
      error("Interfile error determining storage order");
    }

  // can only do this now after the previous things were set
  if (InterfileRawDataHeaderSiemens::post_processing() == true)
    return true;

  return false;
}


END_NAMESPACE_STIR
