// $Id$: $Date$

#include "utilities.h"

// KT 09/10/98 new
PETImageOfVolume ask_image_details()
{
  int scanner_num = 0;
  int span = 1 ;
  PETScannerInfo scanner;

  char filename[256];
  cout << "Enter file name " <<endl;
  cin >> filename;

  // Open file with data
  ifstream input;
  open_read_binary(input, filename);

  scanner_num = ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE)", 0,3,0);
  // span ==1 only supported now
  span = 1;

  switch( scanner_num )
    {
    case 0:
      scanner = (PETScannerInfo::RPT); 
      break;
    case 1:
      scanner = (PETScannerInfo::E953); 
      break;
    case 2:
      scanner = (PETScannerInfo::E966); 
      break;
    case 3:
      scanner = (PETScannerInfo::Advance); 
      break;
    default:
      PETerror("Wrong scanner number\n"); Abort();
    }

  NumericType data_type;
  {
    int data_type_sel = ask_num("Type of data :\n\
0: signed 16bit int, 1: unsigned 16bit int, 2: 4bit float ", 0,2,2);
    switch (data_type_sel)
      { 
      case 0:
	data_type = NumericType::SHORT;
	break;
      case 1:
	data_type = NumericType::USHORT;
	break;
      case 2:
	data_type = NumericType::FLOAT;
	break;
      }
  }


  {
    // find offset 

    input.seekg(0L, ios::beg);   
    unsigned long file_size = find_remaining_size(input);

    unsigned long offset_in_file = ask_num("Offset in file (in bytes)", 
			     0UL,file_size, 0UL);
    input.seekg(offset_in_file, ios::beg);
  }

  
  Point3D origin(0,0,0);
  Point3D voxel_size(scanner.bin_size,
                     scanner.bin_size,
                     scanner.ring_spacing/2); 

  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0 &&
      ask("Make x,y size odd ?", true))
    max_bin++;
   

  PETImageOfVolume 
    input_image(Tensor3D<float>(
				0, 2*scanner.num_rings-2,
				(-scanner.num_bins/2), max_bin,
				(-scanner.num_bins/2), max_bin),
		origin,
		voxel_size);


  Real scale = Real(1);
  input_image.read_data(input, data_type, scale);  
  assert(scale==1);

  return input_image; 

}

PETSinogramOfVolume ask_PSOV_details(iostream * p_in_stream,
				     const bool on_disk)
{

  int scanner_num = 0;
  int span = 1 ;
  PETScannerInfo scanner;

  char filename[256];
  cout << "Enter file name " <<endl;
  cin >> filename;

  if (on_disk)
    {
    
      fstream * p_fstream = new fstream;
      open_read_binary(*p_fstream, filename);
      p_in_stream = p_fstream;
    }
  else
    {  
      unsigned long file_size = 0;
      char *memory = 0;
      { 
	fstream input;
	open_read_binary(input, filename);
	memory = (char *)read_stream_in_memory(input, file_size);
      }
    
#ifdef __GNUG__
      // This is the old implementation of the strstream class.
      // The next constructor should work according to the doc, but it doesn't in gcc 2.8.1.
      //strstream in_stream(memory, file_size, ios::in | ios::binary);
      // Reason: in_stream contains an internal strstreambuf which is 
      // initialised as buffer(memory, file_size, memory), which prevents
      // reading from it.
    
      strstreambuf * buffer = new strstreambuf(memory, file_size, memory+file_size);
      p_in_stream = new iostream(buffer);
#else
      // TODO this does allocate and copy 2 times
      p_in_stream = new stringstream (string(memory, file_size), 
				      ios::in | ios::binary);
      delete[] memory;
#endif
    
    } // else 'on_disk' 
 
 

  scanner_num = ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE)", 0,3,0);
  // span ==1 only supported now
  span = 1;

  switch( scanner_num )
    {
    case 0:
      scanner = (PETScannerInfo::RPT); 
      break;
    case 1:
      scanner = (PETScannerInfo::E953); 
      break;
    case 2:
      scanner = (PETScannerInfo::E966); 
      break;
    case 3:
      scanner = (PETScannerInfo::Advance); 
      break;
    default:
      PETerror("Wrong scanner number\n"); Abort();
    }
  scanner.show_params();

  int max_delta = ask_num("Max. ring difference acquired",
			  0,
			  scanner.num_rings-1,
			  scanner.type == PETScannerInfo::Advance 
			  ? 11 : scanner.num_rings-1);

  PETSinogramOfVolume::StorageOrder storage_order;
  {
    int data_org = ask_num("Type of data organisation:\n\
0: SegmentRingViewBin, 1: SegmentViewRingBin, 2: ViewSegmentRingBin ", 
			   0,
			   2,
			   scanner.type == PETScannerInfo::Advance ? 2 : 0);

    switch (data_org)
      { 
      case 0:
	storage_order = PETSinogramOfVolume::SegmentRingViewBin;
	break;
      case 1:
	storage_order = PETSinogramOfVolume::SegmentViewRingBin;
	break;
      case 2:
	storage_order = PETSinogramOfVolume::ViewSegmentRingBin;
	break;
      }
  }

  NumericType data_type;
  {
    int data_type_sel = ask_num("Type of data :\n\
0: signed 16bit int, 1: unsigned 16bit int, 2: 4bit float ", 0,2,0);
    switch (data_type_sel)
      { 
      case 0:
	data_type = NumericType::SHORT;
	break;
      case 1:
	data_type = NumericType::USHORT;
	break;
      case 2:
	data_type = NumericType::FLOAT;
	break;
      }
  }

  long offset_in_file ;
  {
    // find file size
    p_in_stream->seekg(0L, ios::beg);   
    unsigned long file_size = find_remaining_size(*p_in_stream);

    offset_in_file = ask_num("Offset in file (in bytes)", 
			     0UL,file_size, 0UL);
  }

  return PETSinogramOfVolume(scanner, span, max_delta,
			     *p_in_stream, offset_in_file,
			     storage_order,
			     data_type,
			     Real(1));

}

// find number of remaining characters
streamsize find_remaining_size (istream& input)
{
   streampos file_current_pos = input.tellg();
   input.seekg(0L, ios::end);
   streampos file_end = input.tellg();
   input.clear(); // necessary because seek past EOF ?
   input.seekg(file_current_pos);
   return file_end - file_current_pos;
}

void * read_stream_in_memory(istream& input, unsigned long& file_size)
{
  if (file_size == 0)
    file_size = find_remaining_size(input);
 
  cerr << "Reading " << file_size << " bytes from file." <<endl;

  // allocate memory
  char *memory = new char[file_size];
  if (memory == 0)
    { PETerror("Not enough memory\n"); Abort(); }

  {
    const unsigned long chunk_size = 1024*64;
    unsigned long to_read = file_size;
    char *current_location = memory;

    while( to_read != 0)
      {
	const unsigned long this_read_size = min(to_read, chunk_size);
	input.read(current_location, this_read_size);
	if (!input)
	  { PETerror("Error after reading from stream\n"); Abort(); }

	to_read -= this_read_size;
	current_location += this_read_size;
      }
  }
  return memory;
}
