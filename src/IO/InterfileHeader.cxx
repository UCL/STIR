//
// $Id$: $Date$
//
/*!
  \file 
  \ingroup buildblock 
  \brief implementations for the InterfileHeader class

  \author Kris Thielemans
  \author PARAPET project

  \date   $Date$
  \version $Revision$

*/

#include "InterfileHeader.h"
#include <numeric>

#ifndef TOMO_NO_NAMESPACES
using std::binary_function;
using std::pair;
using std::sort;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_TOMO
InterfileHeader::InterfileHeader()
     : KeyParser()
{

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


// MJ 17/05/2000 made bool
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
    warning("Interfile Warning: only 'type of data := PET' supported.\n");

  // KT 29/10/98 new
  if (matrix_size.size()==0 || matrix_size[matrix_size.size()-1].size()!=1)
  {
    warning("Interfile error: matrix size keywords not in expected format\n");
    return true;
  }

  for (int frame=0; frame<num_time_frames; frame++)
  {
    if (image_scaling_factors[frame].size() == 1)
    {
      // use the only value for every scaling factor
      image_scaling_factors[frame].resize(matrix_size[matrix_size.size()-1][0]);
      for (unsigned int i=1; i<image_scaling_factors[frame].size(); i++)
	image_scaling_factors[frame][i] = image_scaling_factors[frame][0];
    } 
    else if (static_cast<int>(image_scaling_factors[frame].size()) !=  matrix_size[matrix_size.size()-1][0])
    {
      warning("Interfile error: wrong number of image scaling factors\n");
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

// MJ 17/05/2000 made bool
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
      warning("Interfile: only supporting x,y,z order of coordinates now.\n");
      return true; 
    }


  return false;
}
/**********************************************************************/

//KT 26/10/98
// KT 13/11/98 moved stream arg from constructor to parse()
InterfilePDFSHeader::InterfilePDFSHeader()
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
    (KeywordProcessor)&InterfilePDFSHeader::resize_segments_and_set, 
    &min_ring_difference);
  add_key("maximum ring difference per segment",
    KeyArgument::LIST_OF_INTS, 
    (KeywordProcessor)&InterfilePDFSHeader::resize_segments_and_set, 
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

void InterfilePDFSHeader::resize_segments_and_set()
{
  // find_storage_order returns true if already found (or error)
  if (num_segments < 0 && !find_storage_order())
  {
    min_ring_difference.resize(num_segments);
    max_ring_difference.resize(num_segments);
    
  }
  
  if (num_segments >= 0)
    set_variable();
  
}

int InterfilePDFSHeader::find_storage_order()
{
  
  if (num_dimensions != 4)
  { 
    warning("Interfile error: expecting 4D structure "); 
    stop_parsing();
    return true; 
  }
  
  if (matrix_labels[0] != "tangential coordinate")
  { 
    // use error message with index [1] as that is what the user sees.
    warning("Interfile error: expecting 'matrix axis label[1] := tangential coordinate'\n"); 
    stop_parsing();
    return true; 
  }
  num_bins = matrix_size[0][0];
  
  if (matrix_labels[3] == "segment")
  {
    num_segments = matrix_size[3][0];
    
    if (matrix_labels[1] == "axial coordinate" && matrix_labels[2] == "view")
    {
      storage_order =ProjDataFromStream::Segment_View_AxialPos_TangPos;
      num_views = matrix_size[2][0];
#ifdef _MSC_VER
      num_rings_per_segment.assign(matrix_size[1].begin(), matrix_size[1].end());
#else      
      num_rings_per_segment = matrix_size[1];
#endif
    }
    else if (matrix_labels[1] == "view" && matrix_labels[2] == "axial coordinate")
    {
      storage_order = ProjDataFromStream::Segment_AxialPos_View_TangPos;
      num_views = matrix_size[1][0];
#ifdef _MSC_VER
      
      num_rings_per_segment.assign(matrix_size[2].begin(), matrix_size[2].end());
      
#else
      num_rings_per_segment = matrix_size[2];
#endif
    }
    
  }
  /*
  else if (matrix_labels[3] == "view" && 
  matrix_labels[2] == "segment" && matrix_labels[1] == "axial coordinate")
  {
  storage_order = ProjDataFromStream::View_Segment_AxialPos_TangPos;
  num_segments = matrix_size[2][0];
  num_views = matrix_size[3][0];
  #ifdef _MSC_VER
  num_rings_per_segment.assign(matrix_size[1].begin(), matrix_size[1].end());
  #else
  num_rings_per_segment = matrix_size[1];
  #endif
  
   }
  */
  else
  { 
    warning("Interfile error: matrix labels not in expected (or supported) format\n"); 
    stop_parsing();
    return true; 
  }
  
  return false;
  
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
void
find_segment_sequence(vector<int>& segment_sequence,
                      VectorWithOffset<int>& sorted_num_rings_per_segment,
		      VectorWithOffset<int>& sorted_min_ring_diff,
		      VectorWithOffset<int>& sorted_max_ring_diff,
		      vector<int>& num_rings_per_segment,
		      const vector<int>& min_ring_difference, 
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
#if 0
  cerr<< "DISPLAY SUM and LOCATION\n"<<endl;
  
  cerr<<"SUM\n"<<endl;
  for(unsigned int i = 0;i<sum_and_location.size();i++)
  {
    cerr<< sum_and_location[i].first<<" ";
  }
  cerr<<endl;
  
  cerr<<"Location\n"<<endl;
  for(unsigned int i = 0;i<sum_and_location.size();i++)
  {
    cerr<< sum_and_location[i].second<<" ";
  }
  cerr<<endl;
#endif  
  
  // sort with respect to 'sum'
  std::sort(sum_and_location.begin(), sum_and_location.end(),  
    compare_first<float, int>());
#if 0  
  cerr<<"display  sum_sorted"<<endl;
  for(unsigned int i = 0;i<sum_and_location.size();i++)
  {
    cerr<< sum_and_location[i].first<<" ";
  }
  cerr<<endl;
#endif  
  
  
  // find number of segment 0
  int segment_zero_num = 0;
  while (segment_zero_num < num_segments &&
    sum_and_location[segment_zero_num].first < -1E-3)
    segment_zero_num++;
  
  if (segment_zero_num == num_segments ||
    sum_and_location[segment_zero_num].first > 1E-3)
  {
  error("This data does not seem to contain segment 0. \n\
    We can't handle this at the moment. Sorry.");
  }
  
  vector< pair<int, int> > location_and_segment_num(num_segments);
  for (int i=0; i<num_segments; i++)
  {
    location_and_segment_num[i].first = sum_and_location[i].second;
    location_and_segment_num[i].second = i - segment_zero_num;
  }

#if 0
  cerr<< "display location segment\n"<<endl;
  for(unsigned int i = 0;i<location_and_segment_num.size();i++)
  {
    cerr<< location_and_segment_num[i].first<<" ";
  }
  cerr<<endl;
  
  cerr<< "display segment\n"<<endl;
  for(unsigned int i = 0;i<location_and_segment_num.size();i++)
  {
    cerr<< location_and_segment_num[i].second<<" ";
  }
  cerr<<endl;
#endif
  
  const int min_segment_num = location_and_segment_num[0].second;
  const int max_segment_num = location_and_segment_num[num_segments-1].second;

  // KT 19/05/2000 replaced limit with min/max_segment_num
  //int limit = static_cast<int>(ceil(num_segments/2 ));
  
  sorted_min_ring_diff = VectorWithOffset<int>(min_segment_num,max_segment_num);
  sorted_max_ring_diff = VectorWithOffset<int>(min_segment_num,max_segment_num);
  sorted_num_rings_per_segment= VectorWithOffset<int>(min_segment_num,max_segment_num);
  
  
  for (int i=0; i<num_segments; i++)
  {
    sorted_min_ring_diff[(location_and_segment_num[i].second)]
      = min_ring_difference[(location_and_segment_num[i].first)];
    
    sorted_max_ring_diff[(location_and_segment_num[i].second)]
      = max_ring_difference[(location_and_segment_num[i].first)];
    
    sorted_num_rings_per_segment[(location_and_segment_num[i].second)]
      = num_rings_per_segment[(location_and_segment_num[i].first)];
    
    
  }

#if 0
  cerr<< "sorted_min_ring_diff\n"<<endl;
  for( int i =min_segment_num;i<max_segment_num;i++)
  {
    cerr<< sorted_min_ring_diff[i]<<" ";
  }

  cerr<<endl;

  cerr<< "sorted_max_ring_diff\n"<<endl;
  for( int i =min_segment_num;i<max_segment_num;i++)
  {
    cerr<< sorted_max_ring_diff[i]<<" ";
  }
  cerr<<endl;

  
  cerr<< "sorted_num_rings_per_segment\n"<<endl;
  for( int i =min_segment_num;i<max_segment_num;i++)
  {
    cerr<< sorted_num_rings_per_segment[i]<<" ";
  }
  cerr<<endl;
#endif

 
  // sort back to original location
  sort(location_and_segment_num.begin(), location_and_segment_num.end(),  
      compare_first<int, int>());
   

   segment_sequence.resize(num_segments);  
    for (int i=0; i<num_segments; i++)
      segment_sequence[i] = location_and_segment_num[i].second;
  
#if 0    
  cerr<< "segment sequence\n"<<endl;
  for(unsigned int i =0;i<segment_sequence.size();i++)
  {
    cerr<< segment_sequence[i]<<" ";
  }
  cerr<<endl;
#endif

    
  //}
	 
	  
}	  
	  
// MJ 17/05/2000 made bool
bool InterfilePDFSHeader::post_processing()
{
  
  if (InterfileHeader::post_processing() == true)
    return true;
  
  if (PET_data_type_values[PET_data_type_index] != "Emission")
  { warning("Interfile error: expecting emission data\n");  return true; }
  
  if (min_ring_difference.size()!= static_cast<unsigned int>(num_segments))
  { 
    warning("Interfile error: per-segment information is inconsistent: min_ring_difference\n"); 
    return true;
  }
  if (max_ring_difference.size() != static_cast<unsigned int>(num_segments))
  { 
    warning("Interfile error: per-segment information is inconsistent: max_ring_difference\n"); 
    return true;
  }
  if (num_rings_per_segment.size()!= static_cast<unsigned int>(num_segments))
  { 
    warning("Interfile error: per-segment information is inconsistent: num_rings_per_segment\n"); 
    return true;
  }
  
  
 
  VectorWithOffset<int> sorted_min_ring_diff;
  VectorWithOffset<int> sorted_max_ring_diff;
  VectorWithOffset<int> sorted_num_rings_per_segment;
  
  find_segment_sequence( segment_sequence,sorted_num_rings_per_segment,
    sorted_min_ring_diff,sorted_max_ring_diff,
    num_rings_per_segment,
    min_ring_difference, max_ring_difference);
  
  cerr << "PDFS data read inferred header :\n";
  cerr << "Segment sequence :";
  for (unsigned int i=0; i<segment_sequence.size(); i++)
    cerr << segment_sequence[i] << "  ";
  cerr << endl;
  cerr << "RingDiff minimum :";
  for (int i=sorted_min_ring_diff.get_min_index(); i<=sorted_min_ring_diff.get_max_index(); i++)
    cerr <<sorted_min_ring_diff[i] << "  ";  cerr << endl;
  cerr << "RingDiff maximum :";
  for (int i=sorted_max_ring_diff.get_min_index(); i<=sorted_max_ring_diff.get_max_index(); i++)
    cerr << sorted_max_ring_diff[i] << "  ";  cerr << endl;
  cerr << "Nbplanes/segment :";
  for (int i=sorted_num_rings_per_segment.get_min_index(); i<=sorted_num_rings_per_segment.get_max_index(); i++)
    cerr << sorted_num_rings_per_segment[i] << "  ";  cerr << endl;

  cerr << "Total number of planes :" 
    << 
#ifndef TOMO_NO_NAMESPACES // stupid work-around for VC
    std::accumulate
#else
    accumulate
#endif
       (num_rings_per_segment.begin(), num_rings_per_segment.end(), 0)
    << endl;
  
  // handle scanner

  Scanner * guessed_scanner_ptr = Scanner::get_scanner_from_name(originating_system);
  if (guessed_scanner_ptr->get_type() == Scanner::Unknown_Scanner)
  {
    // attempt to guess the system by checking the num_views

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
    case 192*2:
      guessed_scanner_ptr = new Scanner( Scanner::E953);
      warning(warning_msg, "ECAT 953");
      break;
    case 336*2:
      guessed_scanner_ptr = new Scanner( Scanner::Advance);
      warning(warning_msg, "Advance");
      break;
    case 288*2:
      guessed_scanner_ptr = new Scanner( Scanner::E966);
      warning(warning_msg, "ECAT 966");
      // KT 10/01/2000 added
    case 256*2:
      guessed_scanner_ptr = new Scanner( Scanner::E951);
      warning(warning_msg, "ECAT 951");
      break;
    default:
    warning("\nInterfile warning: I did not recognise the scanner neither from \n\
      originating_system or 'number of detectors per ring'.\n");;
    guessed_scanner_ptr = new Scanner( Scanner::Unknown_Scanner);
    break;
    }
    
  }
 
  if (guessed_scanner_ptr->get_type() == Scanner::Unknown_Scanner)
  {
    // warn if the Interfile header does not provide enough info

    if (num_rings < 1)
      warning("Interfile warning: 'number of rings' invalid.\n");
    if (num_detectors_per_ring < 1)
      warning("Interfile warning: 'num_detectors_per_ring' invalid.\n");
#if 0
    if (transaxial_FOV_diameter_in_cm < 0)
      warning("Interfile warning: 'transaxial FOV diameter (cm)' invalid.\n");
#endif
    if (ring_diameter_in_cm < 0)
      warning("Interfile warning: 'ring diameter (cm)' invalid.\n");
    if (distance_between_rings_in_cm < 0)
      warning("Interfile warning: 'distance between rings (cm)' invalid.\n");
    if (bin_size_in_cm < 0)
      warning("Interfile warning: 'bin size (cm)' invalid.\n");
  }

  else
  {
     // fill in values which are not in the Interfile header
    
    if (num_rings < 1)
      num_rings = guessed_scanner_ptr->get_num_rings();
    if (num_detectors_per_ring < 1)
      num_detectors_per_ring = guessed_scanner_ptr->get_max_num_views()*2;
    // KT&SM 26/01/2000 compare with 0 instead of 1 in the next few checks
#if 0
    if (transaxial_FOV_diameter_in_cm < 0)
      transaxial_FOV_diameter_in_cm = guessed_scanner_ptr->FOV_radius*2/10.;
#endif
    if (ring_diameter_in_cm < 0)
      ring_diameter_in_cm = guessed_scanner_ptr->get_ring_radius()*2/10.;
    if (distance_between_rings_in_cm < 0)
      distance_between_rings_in_cm = guessed_scanner_ptr->get_ring_spacing()/10;
    if (bin_size_in_cm < 0)
      bin_size_in_cm = guessed_scanner_ptr->get_default_num_arccorrected_bins()/10;
    
 
    // consistency check with values of the guessed_scanner_ptr we guessed above

    const double tolerance = 10E-4;
    if (num_rings != guessed_scanner_ptr->get_num_rings())
      warning("Interfile warning: 'number of rings' (%d) is expected to be %d\n",
      num_rings, guessed_scanner_ptr->get_num_rings());
    if (num_detectors_per_ring != guessed_scanner_ptr->get_max_num_views()*2)
      warning("Interfile warning: 'number of detectors per ring' (%d) is expected to be %d\n",
      num_detectors_per_ring, guessed_scanner_ptr->get_max_num_views()*2);
    if (fabs(ring_diameter_in_cm-guessed_scanner_ptr->get_ring_radius()*2/10.) > tolerance)
      warning("Interfile warning: 'ring diameter (cm)' (%f) is expected to be %f.\n",
      ring_diameter_in_cm, guessed_scanner_ptr->get_ring_radius()*2/10.);
    if (fabs(distance_between_rings_in_cm-guessed_scanner_ptr->get_ring_spacing()/10) > tolerance)
      warning("Interfile warning: 'distance between rings (cm)' (%f) is expected to be %f.\n",
      distance_between_rings_in_cm, guessed_scanner_ptr->get_ring_spacing()/10);
    if (fabs(bin_size_in_cm-guessed_scanner_ptr->get_default_bin_size()/10) > tolerance)
      warning("Interfile warning: 'bin size (cm)' (%f) is expected to be %f.\n",
      bin_size_in_cm, guessed_scanner_ptr->get_default_bin_size()/10);

    }

  // finally, we construct a new scanner object with
  // data from the Interfile header (or the guessed scanner).
  Scanner * scanner_ptr_from_file =
    new Scanner(guessed_scanner_ptr->get_type(), originating_system,
           num_detectors_per_ring, num_rings, 
	   guessed_scanner_ptr->get_max_num_non_arccorrected_bins(), 
	   guessed_scanner_ptr->get_default_num_arccorrected_bins(),
	   ring_diameter_in_cm*10.F/2,
	   distance_between_rings_in_cm*10.F,
	   bin_size_in_cm*10.F,
	   // TODO have keyword
	   guessed_scanner_ptr->get_default_intrinsic_tilt());


  delete guessed_scanner_ptr;
 
  
  // float azimuthal_angle_sampling =_PI/num_views;
  
  
   
  data_info_ptr = new ProjDataInfoCylindricalArcCorr (
    scanner_ptr_from_file,
    float(bin_size_in_cm*10.),
    sorted_num_rings_per_segment,
    sorted_min_ring_diff,
    sorted_max_ring_diff,
    num_views,num_bins);

  
  
  
  cerr << data_info_ptr->parameter_info() << endl;
  
  return false;
}

END_NAMESPACE_TOMO
