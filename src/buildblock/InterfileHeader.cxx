//
// $Id$
//
/*!
  \file 
 
  \brief implementations for the InterfileHeader class

  \author Kris Thielemans
  \author PARAPET project

  \date    00/03/08

  \version 1.10

*/

#include "InterfileHeader.h"
// for accumulate
#include <numeric>

START_NAMESPACE_TOMO

// KT 03/03/2000 changed PETerror to warning

// KT 26/10/98 changed from init_keys
// KT 13/11/98 moved stream arg from constructor to parse()
InterfileHeader::InterfileHeader()
     : KeyParser()
{

  // KT 20/06/98 new, unfortunate syntax...
  number_format_values.push_back("bit");
  number_format_values.push_back("ascii");
  number_format_values.push_back("signed integer");
  number_format_values.push_back("unsigned integer");
  number_format_values.push_back("float");
  
  // KT 01/08/98 new
  byte_order_values.push_back("LITTLEENDIAN");
  byte_order_values.push_back("BIGENDIAN");
  
  PET_data_type_values.push_back("Emission");
  PET_data_type_values.push_back("Transmission");
  PET_data_type_values.push_back("Blank");
  PET_data_type_values.push_back("AttenuationCorrection");
  PET_data_type_values.push_back("Normalisation");
  PET_data_type_values.push_back("Image");
  
  type_of_data_values.push_back("Static");
  type_of_data_values.push_back("Dynamic");
  type_of_data_values.push_back("Tomographic");
  type_of_data_values.push_back("Curve");
  type_of_data_values.push_back("ROI");
  type_of_data_values.push_back("PET");
  type_of_data_values.push_back("Other");
  
  // default values
  // KT 02/11/98 set default for correct variable
  byte_order_index = 1;//  file_byte_order = ByteOrder::big_endian;
  type_of_data_index = 6; // PET
  // KT 02/11/98 new
  PET_data_type_index = 5; // Image
  // KT 19/10/98 added new ones
  //KT 26/10/98 removed in_stream = 0;
  num_dimensions = 0;
  num_time_frames = 1;
  image_scaling_factors.resize(num_time_frames);
  // KT 29/10/98 factor->factors and new defaults
  for (int i=0; i<num_time_frames; i++)
    image_scaling_factors[i].resize(1, 1.);
  data_offset.resize(num_time_frames, 0UL);


  // KT 09/10/98 replaced NULL arguments with the do_nothing function
  // as gcc cannot convert 0 to a 'pointer to member function'  
  add_key("INTERFILE", 
    KeyArgument::NONE,	&KeyParser::start_parsing);
  // KT 01/08/98 just set data_file_name variable now
  add_key("name of data file", 
    KeyArgument::ASCII,	&data_file_name);
  // KT 26/11/98 new
  add_key("originating system",
    KeyArgument::ASCII, &originating_system);
  add_key("GENERAL DATA", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("GENERAL IMAGE DATA", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("type of data", 
    KeyArgument::ASCIIlist,
    &type_of_data_index, 
    &type_of_data_values);
  
  // KT 08/10/98 removed space in keyword
  add_key("imagedata byte order", 
    KeyArgument::ASCIIlist,
    &byte_order_index, 
    &byte_order_values);
  add_key("PET STUDY (General)", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("PET data type", 
    KeyArgument::ASCIIlist,
    &PET_data_type_index, 
    &PET_data_type_values);
  
  add_key("data format", 
    KeyArgument::ASCII,	&KeyParser::do_nothing);
  // KT 20/06/98 use ASCIIlist
  add_key("number format", 
    KeyArgument::ASCIIlist,
    &number_format_index,
    &number_format_values);
  add_key("number of bytes per pixel", 
    KeyArgument::INT,	&bytes_per_pixel);
  add_key("number of dimensions", 
    KeyArgument::INT,	(KeywordProcessor)&InterfileHeader::read_matrix_info,&num_dimensions);
  add_key("matrix size", 
    KeyArgument::LIST_OF_INTS,&matrix_size);
  add_key("matrix axis label", 
    KeyArgument::ASCII,	&matrix_labels);
  add_key("scaling factor (mm/pixel)", 
    KeyArgument::DOUBLE, &pixel_sizes);
  add_key("number of time frames", 
    KeyArgument::INT,	(KeywordProcessor)&InterfileHeader::read_frames_info,&num_time_frames);
  add_key("PET STUDY (Emission data)", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("PET STUDY (Image data)", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  // TODO
  add_key("process status", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("IMAGE DATA DESCRIPTION", 
    KeyArgument::NONE,	&KeyParser::do_nothing);

  // KT 03/11/98 2 new to avoid error messages
  //TODO
  add_key("maximum pixel count", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("minimum pixel count", 
    KeyArgument::NONE,	&KeyParser::do_nothing);

  // KT 03/11/98 factor -> factors
  add_key("image scaling factor", 
    KeyArgument::LIST_OF_DOUBLES, &image_scaling_factors);

  add_key("data offset in bytes", 
    KeyArgument::ULONG,	&data_offset);
  add_key("END OF INTERFILE", 
    KeyArgument::NONE,	&KeyParser::stop_parsing);
}


bool InterfileHeader::post_processing()
{
    // KT 20/06/98 new
  type_of_numbers = NumericType(number_format_values[number_format_index], bytes_per_pixel);
  
  // KT 01/08/98 new
  file_byte_order = byte_order_index==0 ? 
    ByteOrder::little_endian : ByteOrder::big_endian;
  
  //KT 26/10/98 removed
  //in_stream = new fstream;
  //open_read_binary(*in_stream, data_file_name.c_str());
  if(type_of_data_values[type_of_data_index] != "PET")
    cerr << "Interfile Warning: only 'type of data := PET' supported." << endl;

  // KT 29/10/98 new
  if (matrix_size.size()==0 || matrix_size[matrix_size.size()-1].size()!=1)
  {
    cerr << "Interfile error: matrix size keywords not in expected format" << endl;
    return true;
  }

  for (int frame=0; frame<num_time_frames; frame++)
  {
    if (image_scaling_factors[frame].size() == 1)
    {
      // use the only value for every scaling factor
      image_scaling_factors[frame].resize(matrix_size[matrix_size.size()-1][0]);
      for (int i=1; i<image_scaling_factors[frame].size(); i++)
	image_scaling_factors[frame][i] = image_scaling_factors[frame][0];
    } 
    else if (image_scaling_factors[frame].size() != 
      matrix_size[matrix_size.size()-1][0])
    {
      cerr << "Interfile error: wrong number of image scaling factors" << endl;
      return true;
    }
  }
  
  return false;

}

void InterfileHeader::read_matrix_info()
{
  set_variable();
  matrix_labels.resize(num_dimensions);
  matrix_size.resize(num_dimensions);
  // KT 19/10/98 added default values
  pixel_sizes.resize(num_dimensions, 1.);
  
}

void InterfileHeader::read_frames_info()
{
  set_variable();
  // KT 19/10/98 added default values
  // KT 29/10/98 factor->factors and new defaults
  image_scaling_factors.resize(num_time_frames);
  for (int i=0; i<num_time_frames; i++)
    image_scaling_factors[i].resize(1, 1.);
  data_offset.resize(num_time_frames, 0UL);
  
}

/***********************************************************************/

bool InterfileImageHeader::post_processing()
{

  if (InterfileHeader::post_processing() == true)
    return true;

  if (PET_data_type_values[PET_data_type_index] != "Image")
    { warning("Interfile error: expecting an image\n");  return true; }
  
  if (num_dimensions != 3)
    { warning("Interfile error: expecting 3D image\n"); return true; }

  // KT 29/10/98 moved from asserts in read_interfile_image
  if ( (matrix_size[0].size() != 1) || 
       (matrix_size[1].size() != 1) ||
       (matrix_size[2].size() != 1) )
  { warning("Interfile error: only handling image with homogeneous dimensions\n"); return true; }

  // KT 09/10/98 changed order z,y,x->x,y,z
  // KT 09/10/98 allow no labels at all
  if (matrix_labels[0].length()>0 
      && (matrix_labels[0]!="x" || matrix_labels[1]!="y" ||
	  matrix_labels[2]!="z"))
    {
      cerr << "Interfile: only supporting x,y,z order of coordinates now."
	   << endl; 
      return true; 
    }


  return false;
}
/**********************************************************************/

//KT 26/10/98
// KT 13/11/98 moved stream arg from constructor to parse()
InterfilePSOVHeader::InterfilePSOVHeader()
     : InterfileHeader()
{
  num_segments = -1;

  //KT 26/10/98 changed INT->LIST_OF_INTS
  /* KT 12/11/98 removed
  add_key("segment sequence", 
    KeyArgument::LIST_OF_INTS, 
    (KeywordProcessor)&InterfilePSOVHeader::resize_segments_and_set, 
    &segment_sequence);
    */
  //KT 26/10/98 added 'per segment'
  add_key("minimum ring difference per segment",
    KeyArgument::LIST_OF_INTS, 
    (KeywordProcessor)&InterfilePSOVHeader::resize_segments_and_set, 
    &min_ring_difference);
  add_key("maximum ring difference per segment",
    KeyArgument::LIST_OF_INTS, 
    (KeywordProcessor)&InterfilePSOVHeader::resize_segments_and_set, 
    &max_ring_difference);

  
  // KT 26/11/98 new
  // first set to some crazy values
  num_rings = -1;
  add_key("number of rings",
    KeyArgument::INT, &num_rings);
  num_detectors_per_ring = -1;
  add_key("number of detectors per ring",
    KeyArgument::INT, &num_detectors_per_ring);
  transaxial_FOV_diameter_in_cm = -1;
  add_key("transaxial FOV diameter (cm)",
    KeyArgument::DOUBLE, &transaxial_FOV_diameter_in_cm);
  // KT 31/03/99 new
  ring_diameter_in_cm = -1;
  add_key("ring diameter (cm)",
    KeyArgument::DOUBLE, &ring_diameter_in_cm);
  distance_between_rings_in_cm = -1;
  add_key("distance between rings (cm)",
    KeyArgument::DOUBLE, &distance_between_rings_in_cm);
  bin_size_in_cm = -1;
  add_key("bin size (cm)",
    KeyArgument::DOUBLE, &bin_size_in_cm);
  // this is a good default value
  view_offset = 0;
  add_key("view offset (degrees)",
    KeyArgument::DOUBLE, &view_offset);
}

void InterfilePSOVHeader::resize_segments_and_set()
{
  //KT 26/10/98 find_storage_order returns 1 if already found (or error)
  if (num_segments < 0 && !find_storage_order())
  {
    // KT 12/11/98 removed
    // segment_sequence.resize(num_segments);
    min_ring_difference.resize(num_segments);
    max_ring_difference.resize(num_segments);
  }

  if (num_segments >= 0)
    set_variable();

}

int InterfilePSOVHeader::find_storage_order()
{

  if (num_dimensions != 4)
  { 
    warning("Interfile error: expecting 4D structure "); 
    stop_parsing();
    return 1; 
  }
  
  if (matrix_labels[3] != "bin")
  { 
    warning("Interfile error: expecting 'matrix axis label[4] := bin'\n"); 
    stop_parsing();
    return 1; 
  }
  num_bins = matrix_size[3][0];

  if (matrix_labels[0] == "segment")
  {
    num_segments = matrix_size[0][0];

    if (matrix_labels[1] == "z" && matrix_labels[2] == "view")
    {
      storage_order = PETSinogramOfVolume::SegmentRingViewBin;
      num_views = matrix_size[2][0];
#ifdef _MSC_VER
      num_rings_per_segment.assign(matrix_size[1].begin(), matrix_size[1].end());
#else
      num_rings_per_segment = matrix_size[1];
#endif
    }
    else if (matrix_labels[1] == "view" && matrix_labels[2] == "z")
    {
      storage_order = PETSinogramOfVolume::SegmentViewRingBin;
      num_views = matrix_size[1][0];
#ifdef _MSC_VER
      num_rings_per_segment.assign(matrix_size[2].begin(), matrix_size[2].end());
#else
      num_rings_per_segment = matrix_size[2];
#endif
    }

  }
  else if (matrix_labels[0] == "view" && 
    matrix_labels[1] == "segment" && matrix_labels[2] == "z")
  {
    storage_order = PETSinogramOfVolume::ViewSegmentRingBin;
    num_segments = matrix_size[1][0];
    num_views = matrix_size[0][0];
#ifdef _MSC_VER
    num_rings_per_segment.assign(matrix_size[2].begin(), matrix_size[2].end());
#else
    num_rings_per_segment = matrix_size[2];
#endif
  }
  else
  { 
    warning("Interfile error: matrix labels not in expected (or supported) format\n"); 
    stop_parsing();
    return 1; 
  }

  return 0;

}

#include <functional>

// definition for using sort() below.
// This is a function object that allows comparing the first elements of 2 
// pairs.
template <class T1, class T2>
class compare_first :
        public binary_function<T1, T1, bool> 
{
public:
  bool operator()(const pair<T1, T2>& p1, const pair<T1, T2>& p2)  const
  {
    return p1.first < p2.first;
  }
};


// This function assigns segment numbers by sorting the average 
// ring differences. It returns a list of the segment numbers 
// in the same order as the min/max_ring_difference vectors
vector<int> 
find_segment_sequence(const vector<int>& min_ring_difference, 
		      const vector<int>& max_ring_difference)
{
  const int num_segments = min_ring_difference.size();
  assert(num_segments%2 == 1);

  
  vector< pair<float, int> > sum_and_location(num_segments);
  for (int i=0; i<num_segments; i++)
  {
    sum_and_location[i].first = min_ring_difference[i] + max_ring_difference[i];
    sum_and_location[i].second = i;
  }
  // sort with respect to 'sum'
  sort(sum_and_location.begin(), sum_and_location.end(),  
       compare_first<float, int>());
  

  // find number of segment 0
  int segment_zero_num = 0;
  while (segment_zero_num < num_segments &&
         sum_and_location[segment_zero_num].first < -1E-3)
    segment_zero_num++;

  if (segment_zero_num == num_segments ||
      sum_and_location[segment_zero_num].first > 1E-3)
  {
    warning("This data does not seem to contain segment 0. \n\
 We can't handle this at the moment. Sorry.");
    Abort();
  }

  vector< pair<int, int> > location_and_segment_num(num_segments);
  for (int i=0; i<num_segments; i++)
  {
    location_and_segment_num[i].first = sum_and_location[i].second;
    location_and_segment_num[i].second = i - segment_zero_num;
  }
  
  // sort back to original location
  sort(location_and_segment_num.begin(), location_and_segment_num.end(),  
        compare_first<int, int>());
 
  vector<int> sqc(num_segments);  
  for (int i=0; i<num_segments; i++)
    sqc[i] = location_and_segment_num[i].second;

  
  return sqc;
}

bool InterfilePSOVHeader::post_processing()
{

  if (InterfileHeader::post_processing() == true)
    return true;

  if (PET_data_type_values[PET_data_type_index] != "Emission")
    { warning("Interfile error: expecting emission data\n");  return true; }
  
  // KT 29/10/98 some more checks
  // KT 12/11/98 removed segment_sequence
  if (//segment_sequence.size() != num_segments ||
      min_ring_difference.size() != num_segments ||
      max_ring_difference.size() != num_segments ||
      num_rings_per_segment.size() != num_segments)
    { 
      warning("Interfile error: per-segment information is inconsistent\n"); 
      return true;
    }

  //KT 12/11/98 derived segment_sequence fro ring differences
  // KT 12/11/98 a little bit later, use the class member.
  segment_sequence = 
    find_segment_sequence(min_ring_difference, max_ring_difference);

  cerr << "PSOV data read inferred header :" << endl;
  cerr << "Segment sequence :";
  for (int i=0; i<segment_sequence.size(); i++)
    cerr << segment_sequence[i] << "  ";
  cerr << endl;
  cerr << "RingDiff minimum :";
  for (int i=0; i<min_ring_difference.size(); i++)
    cerr << min_ring_difference[i] << "  ";  cerr << endl;
  cerr << "RingDiff maximum :";
  for (int i=0; i<max_ring_difference.size(); i++)
    cerr << max_ring_difference[i] << "  ";  cerr << endl;
  cerr << "Nbplanes/segment :";
  for (int i=0; i<num_rings_per_segment.size(); i++)
    cerr << num_rings_per_segment[i] << "  ";  cerr << endl;
  cerr << "Total number of planes :" 
#ifndef TOMO_NO_NAMESPACES
       << std::accumulate(num_rings_per_segment.begin(), num_rings_per_segment.end(), 0)
#else
       << accumulate(num_rings_per_segment.begin(), num_rings_per_segment.end(), 0)
#endif
       << endl;


  // Now try to construct a sensible PETScanInfo object.

  // KT 26/11/98 new
  PETScannerInfo::Scanner_type scanner;
  
  // KT 10/01/2000 added 931 and 951. 
  // TODO should come from the PETScannerInfo class
  if (originating_system == "PRT-1")
    scanner = PETScannerInfo::RPT;
  else if (originating_system == "ECAT 931")
    scanner = PETScannerInfo::E931;
  else if (originating_system == "ECAT 951")
    scanner = PETScannerInfo::E951;
  else if (originating_system == "ECAT 953")
    scanner = PETScannerInfo::E953;
  else if (originating_system == "ECAT 966")
    scanner = PETScannerInfo::E966;
  else if (originating_system == "ECAT ART")
    scanner = PETScannerInfo::ART;
  else if (originating_system == "Advance")
    scanner = PETScannerInfo::Advance;
  // SM 22/01/2000 added
  else if (originating_system == "HiDAC")
    scanner = PETScannerInfo::HiDAC;
  // MJ 09/04/2000 added
  else if (originating_system == "Positron HZL/R")
    scanner = PETScannerInfo::HZLR;
  else
  {
    char * warning_msg = 0;
    if (num_detectors_per_ring < 1)
    {
      num_detectors_per_ring = num_views*2;
      warning_msg = "\nInterfile warning: I don't recognise 'originating system' value.\n\
\tI guessed %s from 'num_views' (note: this guess is wrong for mashed data)\n\n";
    }
    else
    {
       warning_msg = "\nInterfile warning: I don't recognise 'originating system' value.\n\
I guessed %s from 'number of detectors per ring'\n";
    }

     
     switch (num_detectors_per_ring)
     {
     case 96*2:
       scanner = PETScannerInfo::RPT;
       warning(warning_msg, "PRT-1");
       break;
     case 192*2:
       scanner = PETScannerInfo::E953;
       warning(warning_msg, "ECAT 953");
       break;
     case 336*2:
       scanner = PETScannerInfo::Advance;
       warning(warning_msg, "Advance");
       break;
     case 288*2:
       scanner = PETScannerInfo::E966;
       warning(warning_msg, "ECAT 966");
     // KT 10/01/2000 added
     case 256*2:
       scanner = PETScannerInfo::E951;
       warning(warning_msg, "ECAT 951");
       break;
     default:
       warning("\nInterfile warning: I did not recognise the scanner neither from \n\
originating_system or 'number of detectors per ring'.\n");;
       scanner = PETScannerInfo::Unknown_Scanner;
       break;
     }
   
  }

  if (scanner == PETScannerInfo::Unknown_Scanner)
  {
    if (num_rings < 1)
      warning("Interfile warning: 'number of rings' invalid.\n");
    if (num_detectors_per_ring < 1)
      warning("Interfile warning: 'num_detectors_per_ring' invalid.\n");
    // KT&SM 26/01/2000 compare with 0 instead of 1 in the next few checks
    // KT 26/01/2000 dropped 'in' from '(in cm)' keywords
    if (transaxial_FOV_diameter_in_cm < 0)
      warning("Interfile warning: 'transaxial FOV diameter (cm)' invalid.\n");
    if (ring_diameter_in_cm < 0)
      warning("Interfile warning: 'ring diameter (cm)' invalid.\n");
    if (distance_between_rings_in_cm < 0)
      warning("Interfile warning: 'distance between rings (cm)' invalid.\n");
    if (bin_size_in_cm < 0)
      warning("Interfile warning: 'bin size (cm)' invalid.\n");
  }
  else
  {
    PETScannerInfo full_scanner(scanner);

   
    if (num_rings < 1)
      num_rings = full_scanner.num_rings;
    if (num_detectors_per_ring < 1)
      num_detectors_per_ring = full_scanner.num_views*2;
    // KT&SM 26/01/2000 compare with 0 instead of 1 in the next few checks
    if (transaxial_FOV_diameter_in_cm < 0)
      transaxial_FOV_diameter_in_cm = full_scanner.FOV_radius*2/10.;
    if (ring_diameter_in_cm < 0)
      ring_diameter_in_cm = full_scanner.ring_radius*2/10.;
    if (distance_between_rings_in_cm < 0)
      distance_between_rings_in_cm = full_scanner.ring_spacing/10;
    if (bin_size_in_cm < 0)
      bin_size_in_cm = full_scanner.bin_size/10;
    

    // consistency check with full_scanner values 
    // KT 26/01/2000 dropped 'in' from '(in cm)' keywords
    // KT 26/01/2000 use tolerance
    const double tolerance = 10E-4;
    if (num_rings != full_scanner.num_rings)
      warning("Interfile warning: 'number of rings' (%d) is expected to be %d\n",
	       num_rings, full_scanner.num_rings);
    if (num_detectors_per_ring != full_scanner.num_views*2)
      warning("Interfile warning: 'number of detectors per ring' (%d) is expected to be %d\n",
	       num_detectors_per_ring, full_scanner.num_views*2);
    if (fabs(transaxial_FOV_diameter_in_cm-full_scanner.FOV_radius*2/10.) > tolerance)
      warning("Interfile warning: 'transaxial FOV diameter (cm)' (%f) is expected to be %f.\n",
		transaxial_FOV_diameter_in_cm, full_scanner.FOV_radius*2/10.);
    if (fabs(ring_diameter_in_cm-full_scanner.ring_radius*2/10.) > tolerance)
      warning("Interfile warning: 'ring diameter (cm)' (%f) is expected to be %f.\n",
		ring_diameter_in_cm, full_scanner.ring_radius*2/10.);
    if (fabs(distance_between_rings_in_cm-full_scanner.ring_spacing/10) > tolerance)
      warning("Interfile warning: 'distance between rings (cm)' (%f) is expected to be %f.\n",
		distance_between_rings_in_cm, full_scanner.ring_spacing/10);
    if (fabs(bin_size_in_cm-full_scanner.bin_size/10) > tolerance)
      warning("Interfile warning: 'bin size (cm)' (%f) is expected to be %f.\n",
	       bin_size_in_cm, full_scanner.bin_size/10);

  }

  scan_info = PETScanInfo(scanner, 
			  num_rings, num_bins, num_views, 
			  float(ring_diameter_in_cm/2*10.),
			  float(distance_between_rings_in_cm*10.), 
			  float(bin_size_in_cm*10.), 
			  float(view_offset));
  
  scan_info.show_params();

  return false;
}

END_NAMESPACE_TOMO
