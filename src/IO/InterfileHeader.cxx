//
// $Id$ :$Date$
//

#include "InterfileHeader.h"

//KT 26/10/98 changed from init_keys
InterfileHeader::InterfileHeader(istream& f)
     : KeyParser(f)
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
  file_byte_order = ByteOrder::big_endian;
  type_of_data_index = 6; // PET
  // KT 19/10/98 added new ones
  //KT 26/10/98 removed in_stream = 0;
  num_dimensions = 0;
  num_time_frames = 1;
  image_scaling_factor.resize(num_time_frames, 1);
  data_offset.resize(num_time_frames, 0);


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
    KeyArgument::DOUBLE,	&pixel_sizes);
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
  add_key("image scaling factor", 
    KeyArgument::DOUBLE,	&image_scaling_factor);
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
  
  return 0;

}

void InterfileHeader::read_matrix_info()
{
  set_variable();
  matrix_labels.resize(num_dimensions);
  matrix_size.resize(num_dimensions);
  // KT 19/10/98 added default values
  pixel_sizes.resize(num_dimensions, 1);
  
}

void InterfileHeader::read_frames_info()
{
  set_variable();
  // KT 19/10/98 added default values
  image_scaling_factor.resize(num_time_frames, 1);
  data_offset.resize(num_time_frames, 0);
  
}

/***********************************************************************/

int InterfileImageHeader::post_processing()
{

  if (InterfileHeader::post_processing() == 1)
    return 1;

  if (PET_data_type_values[PET_data_type_index] != "Image")
    { PETerror("Interfile error: expecting an image");  return 1; }
  
  if (num_dimensions != 3)
    { PETerror("Interfile error: expecting 3D image "); return 1; }
  
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
#if 1
//KT 26/10/98
InterfilePSOVHeader::InterfilePSOVHeader(istream& f)
     : InterfileHeader(f)
{
  num_segments = -1;

  //KT 26/10/98 changed INT->LIST_OF_INTS
  add_key("segment sequence", 
    KeyArgument::LIST_OF_INTS, 
    (KeywordProcessor)&InterfilePSOVHeader::resize_segments_and_set, 
    &segment_sequence);
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
    segment_sequence.resize(num_segments);
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
    PETerror("Interfile error: expecting 'matrix axis label[4] := bin'"); 
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

int InterfilePSOVHeader::post_processing()
{

  if (InterfileHeader::post_processing() == 1)
    return 1;

  if (PET_data_type_values[PET_data_type_index] != "Emission")
    { PETerror("Interfile error: expecting emission data\n");  return 1; }
  
  //KT 26/10/98
  int i;  
  cerr << "Segment sequence :";
  for (i=0; i<segment_sequence.size(); i++)
    cerr << segment_sequence[i] << "  ";
  cerr << endl;
  cerr << "RingDiff minimum :";
  for (i=0; i<min_ring_difference.size(); i++)
    cerr << min_ring_difference[i] << "  ";  cerr << endl;
  cerr << "RingDiff maximum :";
  for (i=0; i<max_ring_difference.size(); i++)
    cerr << max_ring_difference[i] << "  ";  cerr << endl;
  cerr << "Nbplanes/segment :";
  for (i=0; i<num_rings_per_segment.size(); i++)
    cerr << num_rings_per_segment[i] << "  ";  cerr << endl;
  cerr << "Total number of planes :" 
       << accumulate(num_rings_per_segment.begin(), num_rings_per_segment.end(), 0)
       << endl;

  return 0;
}
#endif
