// $Id$

#include "utilities.h"


// KT 21/10/98 new ask_ functions

const char * const 
find_filename(const char * const filename_with_directory)
{
  char *name;

#if __OS__ == __VAX__
 name = strrchr(filename_with_directory,']');
 if (name==NULL)
   name = strrchr(filename_with_directory,':');
#elif __OS__ == __WIN__
 name = strrchr(filename_with_directory,'\\');
 if (name==NULL)
   name = strrchr(filename_with_directory,'/');
 if (name==NULL)
   name = strrchr(filename_with_directory,':');
#elif __OS__ == __MAC__
 name = strrchr(filename_with_directory,':');
#else __OS__ == __UNIX__
 name = strrchr(filename_with_directory,'/');
#endif /* VAX */
 if (name!=NULL)
   return name++;
 else
   return filename_with_directory;
}

char *add_extension(char *file_in_directory_name, 
		    const char * const extension)
{
  if (strchr(find_filename(file_in_directory_name),'.') == NULL)
    strcat (file_in_directory_name,extension);
  return file_in_directory_name;
}

char *ask_filename_with_extension(char *file_in_directory_name,
				  const char * const prompt,
				  const char * const default_extension)
{
  char ptr[max_filename_length];
  
  file_in_directory_name[0]='\0';
  while (strlen(file_in_directory_name)==0)
  { 
    printf ("\n%s ", prompt);
    if (strlen(default_extension))
      printf("(default extension '%s')", default_extension);
    printf(":");
    fgets(ptr,max_filename_length,stdin);
    sscanf(ptr," %s",file_in_directory_name);
  }
  add_extension(file_in_directory_name,default_extension);
  return(file_in_directory_name);
}


template <class FSTREAM>
void
ask_filename_and_open(FSTREAM& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed)
{
  char filename[max_filename_length];
  s.open(
    ask_filename_with_extension(filename, prompt, default_extension),
    mode);
  if (abort_if_failed && !s)
  { PETerror("Error opening file %s\n", filename); Abort(); }
}

// instantiations

template 
void
ask_filename_and_open(ifstream& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);
template 
void
ask_filename_and_open(ofstream& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);
template 
void
ask_filename_and_open(fstream& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);


// KT 09/10/98 new
PETImageOfVolume ask_image_details()
{
 

  // Open file with data
  ifstream input;
  // KT 21/10/98 use new function
  ask_filename_and_open(
    input, "Enter filename for input image", ".v", 
    ios::in | ios::binary);

  int scanner_num = 
      ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE 4: ART)", 0,4,0);// CL 061298 Add ART scan
 
  PETScannerInfo scanner;
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
        case 4:// CL 061298 Add ART scanner
      scanner = (PETScannerInfo::ART); 
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

 
 


  char filename[256];
  cout << endl;
  system("ls *scn *dat *bin");//CL 14/10/98 ADd this printing out of some data files
  
  // KT 21/10/98 use new function
  ask_filename_with_extension(
    filename, 
    "Enter file name of 3D sinogram data : ", ".scn");

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
 
 

  int scanner_num = ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE, 4: ART)", 0,4,0);//CL 290199 Add the ART scanner
  PETScannerInfo scanner;

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
    case 4://CL 290199 ADd the ART scanner
      scanner = (PETScannerInfo::ART); 
      break;

    default:
      PETerror("Wrong scanner number\n"); Abort();
    }

  // KT 15/10/98 allow reading of extended sinograms
 
  if (scanner.num_bins % 2 == 0 &&
      ask("Make num_bins odd ?", false))
    scanner.num_bins++;

  scanner.show_params();

  // KT 19/10/99 change default to 1
  int span = ask_num("Span value : ", 1,11,1);
    
  int max_delta = ask_num("Max. ring difference acquired : ",
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


  ByteOrder byte_order;
  { 
      // KT 19/10/99 change default to false    
    byte_order = ask("Little endian byte order ?", false) ?
      ByteOrder::little_endian :
      ByteOrder::big_endian;
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
			     // KT 15/03/99 new and removed Real(1) (is default)
			     byte_order);

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

