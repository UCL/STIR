#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/Succeeded.h"

#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ios;
using std::streampos;
#endif

#define MAX_STRING_LENGTH 512

START_NAMESPACE_STIR  

Polaris_MT_File::Polaris_MT_File(const string& mt_filename)
{
  char DataStr[MAX_STRING_LENGTH];
  mt_stream.open(mt_filename.c_str(), ios::in);
  if (!mt_stream)
  {
    error( "\n\t\tError Opening Supplied File %s - Does it Exist?\n", mt_filename.c_str()) ;
  }
  
  /* Read opening line - discard */
  if ( !mt_stream.getline( DataStr, MAX_STRING_LENGTH) )
  {
    error("\n\t\tError Reading Line 1 of Supplied File %s\n", mt_filename.c_str());
    mt_stream.close();
  }

  read_mt_file (mt_filename);

} 


#if 1
void
Polaris_MT_File::read_mt_file (const string& filename)
{
  Record record; 
  while (!mt_stream.eof())
  {
    Succeeded record_read=get_next(record);
    if (record_read==Succeeded::yes)
    vector_of_records.push_back(record);
    else
    continue;
  }

}
  
#endif
Succeeded
Polaris_MT_File::get_next(Record& record)
{
  char DataStr[MAX_STRING_LENGTH];  
  mt_stream.getline( DataStr, MAX_STRING_LENGTH);
  
    /* Extract elements from string */
   if (sscanf( DataStr, "%f %d %c %f %f %f %f %f %f %f %f", 
	&record.sample_time, &record.rand_num, &record.total_num, 
	&record.quat[1], &record.quat[2], &record.quat[3], &record.quat[4], 
	&record.trans.x(), &record.trans.y(), &record.trans.z(), 
	&record.rms ) !=11)
    return Succeeded::no;
   else
    return Succeeded::yes;
}


Polaris_MT_File::Record 
Polaris_MT_File::operator[](unsigned int in) const
{
  assert(in>=0);
  return vector_of_records[in];
}

#if 0
ifstream
Polaris_MT_File::get_stream()
{
  return mt_stream;
}
#endif
Succeeded 
Polaris_MT_File::reset()
{
  streampos starting_stream_position;
  mt_stream.seekg(starting_stream_position, ios::beg);

  if (!mt_stream.bad())
  { 
    if (mt_stream.eof()) 
    { 
      // Strangely enough, once you read past EOF, even seekg(0) doesn't reset the eof flag
      mt_stream.clear();
      return Succeeded::yes;
    }
    else
      return Succeeded::no;
  }
  else
  {
    error("Error after seeking to start of data in Polaris_MT_File::reset()\n");      
    return Succeeded::no;
  }
  //return Succeeded::yes;
  
}

Succeeded 
Polaris_MT_File::is_end_file()
{
  if (mt_stream.eof())
    return Succeeded::yes;
  else
    return Succeeded::no;

}

END_NAMESPACE_STIR  

