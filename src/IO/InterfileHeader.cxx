//
// $Id$ :$Date$
//

#include "InterfileHeader.h"

void InterfileHeader::init_keys()
{
  map_element m;
  

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
  in_stream = 0;
  num_dimensions = 0;
  num_time_frames = 1;
  image_scaling_factor.resize(num_time_frames, 1);
  data_offset.resize(num_time_frames, 0);


  // KT 09/10/98 replaced NULL arguments with the DoNothing function
  // as gcc cannot convert 0 to a 'pointer to member function'  
  kmap["INTERFILE"]=			m(KeyArgument::NONE,	&KeyParser::DoNothing);
  // KT 01/08/98 just set data_file_name variable now
  kmap["name of data file"]=		m(KeyArgument::ASCII,	&KeyParser::SetVariable, &data_file_name);
  kmap["GENERAL DATA"]=			m(KeyArgument::NONE,	&KeyParser::DoNothing);
  kmap["GENERAL IMAGE DATA"]=		m(KeyArgument::NONE,	&KeyParser::DoNothing);
  kmap["type of data"]=			m(KeyArgument::ASCIIlist,&KeyParser::SetVariable,
    &type_of_data_index, 
    &type_of_data_values);
  
  // KT 08/10/98 removed space in keyword
  kmap["imagedata byte order"]=		m(KeyArgument::ASCIIlist,&KeyParser::SetVariable,
    &byte_order_index, 
    &byte_order_values);
  kmap["PET STUDY (General)"]=		m(KeyArgument::NONE,	&KeyParser::DoNothing);
  kmap["PET data type"]=		m(KeyArgument::ASCIIlist,&KeyParser::SetVariable,
    &PET_data_type_index, 
    &PET_data_type_values);
  
  kmap["data format"]=			m(KeyArgument::ASCII,	&KeyParser::DoNothing);
  // KT 20/06/98 use ASCIIlist
  kmap["number format"]=		m(KeyArgument::ASCIIlist,&KeyParser::SetVariable,
    &number_format_index,
    &number_format_values);
  kmap["number of bytes per pixel"]=	m(KeyArgument::INT,	&KeyParser::SetVariable,&bytes_per_pixel);
  kmap["number of dimensions"]=		m(KeyArgument::INT,	(KeywordProcessor)&InterfileHeader::ReadMatrixInfo,&num_dimensions);
  kmap["matrix size"]=			m(KeyArgument::LIST_OF_INTS,&KeyParser::SetVariable,&matrix_size);
  kmap["matrix axis label"]=		m(KeyArgument::ASCII,	&KeyParser::SetVariable,&matrix_labels);
  kmap["scaling factor (mm/pixel)"]=	m(KeyArgument::DOUBLE,	&KeyParser::SetVariable,&pixel_sizes);
  kmap["number of time frames"]=	m(KeyArgument::INT,	(KeywordProcessor)&InterfileHeader::ReadFramesInfo,&num_time_frames);
  kmap["PET STUDY (Emission data)"]=	m(KeyArgument::NONE,	&KeyParser::DoNothing);
  kmap["PET STUDY (Image data)"]=	m(KeyArgument::NONE,	&KeyParser::DoNothing);
  // TODO
  kmap["process status"] =		m(KeyArgument::NONE,	&KeyParser::DoNothing);
  kmap["IMAGE DATA DESCRIPTION"]=	m(KeyArgument::NONE,	&KeyParser::DoNothing);
  kmap["image scaling factor"]=		m(KeyArgument::DOUBLE,	&KeyParser::SetVariable,&image_scaling_factor);
  kmap["data offset in bytes"]=		m(KeyArgument::ULONG,	&KeyParser::SetVariable,&data_offset);
  kmap["END OF INTERFILE"]=		m(KeyArgument::NONE,	&KeyParser::SetEndStatus);
  
}


int InterfileHeader::post_processing()
{
    // KT 20/06/98 new
  type_of_numbers = NumericType(number_format_values[number_format_index], bytes_per_pixel);
  
  // KT 01/08/98 new
  file_byte_order = byte_order_index==0 ? 
    ByteOrder::little_endian : ByteOrder::big_endian;
  
  // This has to be here to be able to differentiate between
  // binary open or text open (for ASCII type).
  // TODO we don't support ASCII yet.
  in_stream = new fstream;
  open_read_binary(*in_stream, data_file_name.c_str());
  if(type_of_data_values[type_of_data_index] != "PET")
    cerr << "Interfile Warning: only 'type of data := PET' supported." << endl;
  
  return 0;

}

void InterfileHeader::ReadMatrixInfo()
{
  SetVariable();
  matrix_labels.resize(num_dimensions);
  matrix_size.resize(num_dimensions);
  // KT 19/10/98 added default values
  pixel_sizes.resize(num_dimensions, 1);
  
}

void InterfileHeader::ReadFramesInfo()
{
  SetVariable();
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
#if 0
int InterfilePSOVHeader::post_processing()
{

  if (InterfileHeader::post_processing() == 1)
    return 1;

  if (PET_data_type_values[PET_data_type_index] != "Emission")
    { PETerror("Interfile error: expecting emission data");  return 1; }
  
  if (num_dimensions != 4)
    { PETerror("Interfile error: expecting 4D structure "); return 1; }
  
  if (matrix_labels[3] != "bin")
  { PETerror("Interfile error: expecting 'matrix axis label[4] := bin'"); return 1; }

  if (matrix_labels[0] == "segment")
  {
    if (matrix_labels[1] == "z" && matrix_labels[2] == "view")
      storage_order = PETSinogramOfVolume::SegmentRingViewBin;
    else if (matrix_labels[1] == "view" && matrix_labels[2] == "z")
      storage_order = PETSinogramOfVolume::SegmentViewRingBin;
  }
  else if (matrix_labels[0] == "view" && 
    matrix_labels[1] == "segment" && matrix_labels[2] == "z")
     storage_order = PETSinogramOfVolume::ViewSegmentRingBin;
  else
  { PETerror("Interfile error: matrix labels not in expected format"); return 1; }
      
 
  // TODO

  return 0;
}
#endif
