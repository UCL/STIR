//
// $Id$ :$Date$
//

#include "InterfileHeader.h"

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


int InterfileHeader::post_processing()
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
    return 1;
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
      return 1;
    }
  }
  
  return 0;

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

int InterfileImageHeader::post_processing()
{

  if (InterfileHeader::post_processing() == 1)
    return 1;

  if (PET_data_type_values[PET_data_type_index] != "Image")
    { PETerror("Interfile error: expecting an image\n");  return 1; }
  
  if (num_dimensions != 3)
    { PETerror("Interfile error: expecting 3D image\n"); return 1; }

  // KT 29/10/98 moved from asserts in read_interfile_image
  if ( (matrix_size[0].size() != 1) || 
       (matrix_size[1].size() != 1) ||
       (matrix_size[2].size() != 1) )
  { PETerror("Interfile error: only handling image with homogeneous dimensions\n"); return 1; }

  // KT 09/10/98 changed order z,y,x->x,y,z
  // KT 09/10/98 allow no labels at all
  if (matrix_labels[0].length()>0 
      && (matrix_labels[0]!="x" || matrix_labels[1]!="y" ||
	  matrix_labels[2]!="z"))
    {
      cerr << "Interfile: only supporting x,y,z order of coordinates now."
	   << endl; 
      return 1; 
    }


  return 0;
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
    PETerror("Interfile error: expecting 4D structure "); 
    stop_parsing();
    return 1; 
  }
  
  if (matrix_labels[3] != "bin")
  { 
    PETerror("Interfile error: expecting 'matrix axis label[4] := bin'\n"); 
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
    PETerror("Interfile error: matrix labels not in expected (or supported) format\n"); 
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
    PETerror("This data does not seem to contain segment 0. \n\
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

int InterfilePSOVHeader::post_processing()
{

  if (InterfileHeader::post_processing() == 1)
    return 1;

  if (PET_data_type_values[PET_data_type_index] != "Emission")
    { PETerror("Interfile error: expecting emission data\n");  return 1; }
  
  // KT 29/10/98 some more checks
  // KT 12/11/98 removed segment_sequence
  if (//segment_sequence.size() != num_segments ||
      min_ring_difference.size() != num_segments ||
      max_ring_difference.size() != num_segments ||
      num_rings_per_segment.size() != num_segments)
    { 
      PETerror("Interfile error: per-segment information is inconsistent\n"); 
      return 1;
    }

  //KT 12/11/98 derived segment_sequence fro ring differences
  // KT 12/11/98 a little bit later, use the class member.
  segment_sequence = 
    find_segment_sequence(min_ring_difference, max_ring_difference);

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
       << accumulate(num_rings_per_segment.begin(), num_rings_per_segment.end(), 0)
       << endl;

  return 0;
}

