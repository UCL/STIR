//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations for class Scanner

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/Scanner.h"
#include "stir/utilities.h"
#include <iostream>
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif


#ifndef STIR_NO_NAMESPACES
using std::cout;
using std::endl;
using std::ends;
using std::cin;
using std::find;
#endif

START_NAMESPACE_STIR

// local convenience functions to make a list of strings
static list<string> 
   string_list(const string&);
static list<string> 
   string_list(const string&, const string&);
static list<string> 
   string_list(const string&, const string&, const string&);
static list<string> 
   string_list(const string&, const string&, const string&, const string&);

   


  
Scanner::Scanner(Type scanner_type)
{
  // NoRings, NoBins, 
  // num_detectors_per_ring,
  // RingRadius,
  // RingSpacing, BinSize, intrTilt
  // before arc-correction, central_bin_size = ring_radius* pi/num_detectors for CTI scanners


  if (scanner_type == E931)
    // KT 25/01/2002 corrected ring_spacing
    set_params(E931,string_list("ECAT 931"),  8, 192,2* 256, 510.0F, 13.5F,  3.129F,   0,0,0,0,0);
  else if (scanner_type == E951)
    set_params(E951,string_list("ECAT 951"), 16, 192,2* 256, 510.0F, 6.75F,  3.129F,   0,0,0,0,0);
  else if (scanner_type == E953)
    set_params(E953,string_list("ECAT 953"), 16, 160,2* 192, 382.5F,  6.75F, 3.129F,   15,0,0,0,0);
  else if (scanner_type == E921)
    set_params(E921,string_list("ECAT 921", "ECAT EXACT", "EXACT"), 24, 192,2* 192, 412.5F, 6.75F, 3.375F, 15.F,0,0,0,0);
  else if (scanner_type == E925)
    set_params(E925,string_list("ECAT 925", "ECAT ART"), 24, 192,2* 192, 412.5F, 6.75F, 3.375F, 15.F,0,0,0,0);
  else if (scanner_type == E961)
    set_params(E961,string_list("ECAT 961", "ECAT HR"), 24, 336,2* 196, 412.0F, 6.25F, 1.650F, 13.F,0,0,0,0);
  else if (scanner_type == E962)
    set_params(E962,string_list("ECAT 962","ECAT HR+"), 32, 288,2* 288, 412.5F, 4.85F, 2.25F, 0.F,3,4,8,8);
  else if (scanner_type == E966)
    set_params(E966,string_list("ECAT EXACT 3D", "EXACT 3D", "ECAT HR++","ECAT 966"), 48, 288,2* 288, 412.5F, 4.850F, 2.250F, 0,2,6,8,8); 
  else if (scanner_type == RPT)
    set_params(RPT,string_list("PRT-1", "RPT"), 16, 128,2*192, 380,  6.75F, 3.1088F,   0,0,0,0,0);
  else if (scanner_type == RATPET)
    set_params(RPT,string_list("RATPET"), 8, 56,2*56, 115,  6.25F, 1.65F,   0,0,0,0,0); // HR block
  else if (scanner_type == Advance)
  {
    // Advance option
    // 283 bins (non-uniform sampling) 
    // 281 bins (uniform sampling)
    set_params(Advance,string_list("GE Advance", "Advance"), 18, 283,281,2*336,469.5F, 8.5F, 1.96F, 0,0,0,0,0);
  }
  else if (scanner_type == HZLR)
    set_params(HZLR,string_list("Positron HZL/R"), 32, 256,2* 192, 780.0F, 5.1875F, 2.F, 0,0,0,0,0);
  else if (scanner_type == HRRT)
    set_params(HRRT,string_list("HRRT"), 104, 288, 2*288, 234.765F, 
			   2.4375F, 1.21875F, 0,0,0,0,0); // added by Dylan Togane
  else if (scanner_type == HiDAC)
    // all of these don't make any sense for the HiDAC
    set_params(HiDAC,string_list("HiDAC"), 0, 0, 0, 0.F, 0.F, 0.F, 0,0,0,0,0);
  else
    { 
      // warning("Unknown scanner type used for initialisation of Scanner\n"); 
      set_params(Unknown_Scanner,string_list("Unknown"), 0, 0, 0, 0.F, 0.F, 0.F, 0,0,0,0,0);
    }
}


void
Scanner::set_params(Type type_v,const list<string>& list_of_names_n,
                  int NoRings_v, 
		  int max_num_non_arccorrected_bins_v,
		  int num_detectors_per_ring_v,
		  float RingRadius_v,
		  float RingSpacing_v,
		  float BinSize_v, float intrTilt_v,
		  int trans_blocks_per_bucket_v, int axial_blocks_per_bucket_v,
		  int axial_crystals_per_block_v,int angular_crystals_per_block_v)

{
  type =type_v;
  list_of_names = list_of_names_n;  
  num_rings =  NoRings_v;
  max_num_non_arccorrected_bins = 
  default_num_arccorrected_bins = max_num_non_arccorrected_bins_v;
  num_detectors_per_ring =num_detectors_per_ring_v;  
  ring_radius =  RingRadius_v;
  ring_spacing = RingSpacing_v;
  bin_size = BinSize_v;
  intrinsic_tilt = intrTilt_v;	
  trans_blocks_per_bucket = trans_blocks_per_bucket_v;
  axial_blocks_per_bucket = axial_blocks_per_bucket_v;
  axial_crystals_per_block= axial_crystals_per_block_v;
  angular_crystals_per_block=angular_crystals_per_block_v;
}

void
Scanner::set_params(Type type_v,const list<string>& list_of_names_n,
                  int NoRings_v, 
		  int max_num_non_arccorrected_bins_v,
		  int default_num_arccorrected_bins_v,
		  int num_detectors_per_ring_v,
		  float RingRadius_v,
		  float RingSpacing_v,
		  float BinSize_v, float intrTilt_v,
		  int trans_blocks_per_bucket_v, int axial_blocks_per_bucket_v,
		  int axial_crystals_per_block_v,int angular_crystals_per_block_v)
{
  type =type_v;
  list_of_names = list_of_names_n;  
  num_rings =  NoRings_v;
  max_num_non_arccorrected_bins =  max_num_non_arccorrected_bins_v;
  default_num_arccorrected_bins = default_num_arccorrected_bins_v;
  num_detectors_per_ring =num_detectors_per_ring_v;    
  ring_radius =  RingRadius_v;
  ring_spacing = RingSpacing_v;
  bin_size = BinSize_v;
  intrinsic_tilt = intrTilt_v;	
  trans_blocks_per_bucket = trans_blocks_per_bucket_v;
  axial_blocks_per_bucket = axial_blocks_per_bucket_v;
  axial_crystals_per_block= axial_crystals_per_block_v;
  angular_crystals_per_block=angular_crystals_per_block_v;
}


Scanner::Scanner(Type type_v,
		 const list<string>& list_of_names_n,
		 int num_detectors_per_ring_v,
		 int NoRings_v,	
		 int max_num_non_arccorrected_bins, float RingRadius_v, 
		 float RingSpacing_v, float BinSize_v, float intrTilt_v,
		 int trans_blocks_per_bucket_v, int axial_blocks_per_bucket_v,
		 int axial_crystals_per_block_v,int angular_crystals_per_block_v)
{
  type =type_v;
  list_of_names = list_of_names_n;        
  num_rings =  NoRings_v;
  max_num_non_arccorrected_bins =  max_num_non_arccorrected_bins;
  default_num_arccorrected_bins =default_num_arccorrected_bins;
  num_detectors_per_ring =num_detectors_per_ring_v;
  ring_radius =  RingRadius_v;
  ring_spacing = RingSpacing_v;
  bin_size = BinSize_v;
  intrinsic_tilt = intrTilt_v;	
  trans_blocks_per_bucket = trans_blocks_per_bucket_v;
  axial_blocks_per_bucket = axial_blocks_per_bucket_v;
  axial_crystals_per_block= axial_crystals_per_block_v;
  angular_crystals_per_block=angular_crystals_per_block_v;
	
}

Scanner::Scanner(Type type_v,const string name,
         int num_detectors_per_ring_v, int NoRings_v, int max_num_non_arccorrected_bins, 
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v,
	 int trans_blocks_per_bucket_v, int axial_blocks_per_bucket_v,
         int axial_crystals_per_block_v,int angular_crystals_per_block_v)
{
  set_params(type_v,string_list(name),NoRings_v,
         max_num_non_arccorrected_bins,
         num_detectors_per_ring_v, 
	 RingRadius_v, RingSpacing_v, 
         BinSize_v, intrTilt_v,trans_blocks_per_bucket_v, axial_blocks_per_bucket_v,
         axial_crystals_per_block_v,angular_crystals_per_block_v);

}


Scanner::Scanner(Type type_v,
		 const list<string>& list_of_names_n,
		 int num_detectors_per_ring_v,
		 int NoRings_v,	
		 int max_num_non_arccorrected_bins,
		 int default_num_arccorrected_bins,
		 float RingRadius_v, 
		 float RingSpacing_v, float BinSize_v, float intrTilt_v,
		 int trans_blocks_per_bucket_v, int axial_blocks_per_bucket_v,
		 int axial_crystals_per_block_v,int angular_crystals_per_block_v)
{
  type =type_v;
  list_of_names = list_of_names_n;        
  num_rings =  NoRings_v;
  max_num_non_arccorrected_bins =  max_num_non_arccorrected_bins;
  default_num_arccorrected_bins =default_num_arccorrected_bins;
  num_detectors_per_ring =num_detectors_per_ring_v;
  ring_radius =  RingRadius_v;
  ring_spacing = RingSpacing_v;
  bin_size = BinSize_v;
  intrinsic_tilt = intrTilt_v;	
  trans_blocks_per_bucket = trans_blocks_per_bucket_v;
  axial_blocks_per_bucket = axial_blocks_per_bucket_v;
  axial_crystals_per_block= axial_crystals_per_block_v;
  angular_crystals_per_block=angular_crystals_per_block_v;

	
}

Scanner::Scanner(Type type_v,const string name,
         int num_detectors_per_ring_v, int NoRings_v, 
	 int max_num_non_arccorrected_bins,
	 int default_num_arccorrected_bins,
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v,
	 int trans_blocks_per_bucket_v, int axial_blocks_per_bucket_v,
         int axial_crystals_per_block_v,int angular_crystals_per_block_v)
{
  set_params(type_v,string_list(name),NoRings_v,
         max_num_non_arccorrected_bins,
	 default_num_arccorrected_bins,
         num_detectors_per_ring_v, 
	 RingRadius_v, RingSpacing_v, 
         BinSize_v, intrTilt_v,trans_blocks_per_bucket_v, axial_blocks_per_bucket_v,
         axial_crystals_per_block_v,angular_crystals_per_block_v);

}

// TODO replace by using boost::floating_point_comparison
bool static close_enough(const double a, const double b)
{
  return fabs(a-b) <= 
#ifndef STIR_NO_NAMESPACES
    std::    // need this explicitly here due to VC 6.0 bug
#endif
    min(fabs(a), fabs(b)) * 10E-4;
}

bool 
Scanner::operator ==(const Scanner& scanner) const
{
// KT 04/02/2003 take floating point rounding into account
return
  (num_rings ==scanner.num_rings)&&
  (max_num_non_arccorrected_bins ==scanner.max_num_non_arccorrected_bins)&&
  (default_num_arccorrected_bins ==scanner.default_num_arccorrected_bins)&&
  (num_detectors_per_ring == scanner.num_detectors_per_ring)&&
  close_enough(ring_radius,scanner.ring_radius) &&
  close_enough(bin_size,scanner.bin_size)&&
  close_enough(intrinsic_tilt,scanner.intrinsic_tilt);
}

const list<string>& 
Scanner::get_all_names() const
{return list_of_names;}


const string&
Scanner::get_name() const
{
  
 return *(list_of_names.begin()); 
    
}

string
Scanner::parameter_info() const
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[10000];
  ostrstream s(str, 10000);
#else
  std::ostringstream s;
#endif
  s<<"Scanner parameters info "<<endl;

  s << "Scanner type := "
    << type << endl;
   s << "List of names: " << list_names() <<endl;     
  s << "Number of Rings                        " <<    num_rings << endl;
  s << "Number detectors per ring              " << get_num_detectors_per_ring() << endl;
  s << "Maximum number of nonarccorrected bins " << 
     max_num_non_arccorrected_bins << endl;
  s << "Default number of arccorrected bins    " <<
     default_num_arccorrected_bins << endl;
  s << "Ring radius                            " << 
    ring_radius << endl;
  s << "Ring spacing                           " << 
    ring_spacing << endl;
  s << "Bin size                               " << 
    bin_size << endl;
  s << "Intrinsic tilt                         " << 
    intrinsic_tilt << endl << endl;
  s << "Transaxial blocks per bucket           " << 
    trans_blocks_per_bucket << endl << endl;
  s << "Axial blocks per bucket		       " << 
    axial_blocks_per_bucket << endl << endl;
  s << "Axial crystals per block               " << 
    axial_crystals_per_block << endl << endl;
  s << "Angular crystals per block             " << 
    angular_crystals_per_block << endl << endl;
  s << ends;

  return s.str();
}

string Scanner::list_names() const
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[3000];
  ostrstream s(str, 3000);
#else
  std::ostringstream s;
#endif

#ifdef _MSC_VER
  // work-around VC bug
  std::
#endif
  list<string>::const_iterator iterator = list_of_names.begin(); 
  s << *iterator;
  ++iterator;
  while(iterator!=list_of_names.end())
  {
    s << " , " << *iterator ;
    ++iterator;
  }
  s << ends;

  return s.str();
}

/************************************************
 static members
 *************************************************/
Scanner* Scanner::ask_parameters() 
{

  cout << list_all_names();

  cout<<"Enter the name of the scanner (exact match!)"<<endl;
  char str [100];
  cin.getline(str,100);
  const string name = str;

  //get the type from the name itself
  Scanner* scanner_ptr = 
    get_scanner_from_name(name);

  if (scanner_ptr->type != Unknown_Scanner)
    return scanner_ptr;

  cout << "I didn't recognise the scanner you entered. I'll ask lots of questions\n";
  
  
  int num_detectors_per_ring = 
    ask_num("Enter number of detectors per ring:",0,2000,128);
  
  int NoRings = 
    ask_num("Enter number of rings :",0,128,16);
  
  int NoBins = 
    ask_num("Enter number of bins: ",0,2000,128);
  
  float RingRadius=
    ask_num("Enter ring radius (in mm): ",0.F,500.F,256.F);
  
  float RingSpacing= 
    ask_num("Enter ring spacing (in mm): ",0.F,10.F,6.75F);
  
  float BinSize= 
    ask_num("Enter bin size (in mm):",0.F,10.F,3.75F);
  float intrTilt=
    ask_num("Enter intrinsic_tilt (in degrees):",0.F,360.F,90.F);
  int TransBlocksPerBucket = 
    ask_num("Enter number of trans blocks per bucket: ",0,10,2);
  int AxialBlocksPerBucket = 
    ask_num("Enter number of axial blocks per bucket: ",0,10,6);
  int AxialCrystalsPerBlock = 
    ask_num("Enter number of axial crystals per block: ",0,10,8);
  int AngularCrystalsPerBlock = 
    ask_num("Enter number of angular crystals per block: ",0,10,8);
   Type type = Unknown_Scanner;
  
  Scanner* scanner =
    new Scanner(type,name,
                num_detectors_per_ring,  NoRings, NoBins, 
                RingRadius, RingSpacing, 
                BinSize,intrTilt,TransBlocksPerBucket,AxialBlocksPerBucket,
		AxialCrystalsPerBlock,AngularCrystalsPerBlock );
  
  return scanner;
}


Scanner *
Scanner::get_scanner_from_name(const string& name)
{ 
  Scanner * scanner_ptr;

  Type type= E931; 
  while (type != Unknown_Scanner)
  {
    scanner_ptr = new Scanner(type);
    const list<string>& list_of_names = scanner_ptr->get_all_names();
    if ( std::find(list_of_names.begin(), list_of_names.end(), name)!= list_of_names.end()) 
      return scanner_ptr;
    
    // we didn't find it yet
    delete scanner_ptr;
    // tricky business to find next type
    int int_type = type;
    ++int_type;
    type = static_cast<Type>(int_type);
  }
  // it's not in the list
  return new Scanner(Unknown_Scanner);
}


string Scanner:: list_all_names()
{
  Scanner * scanner_ptr;
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif

  Type type= E931; 
  while (type != Unknown_Scanner)
  {
    scanner_ptr = new Scanner(type);
    s << scanner_ptr->list_names() << endl;
    
    delete scanner_ptr;
    // tricky business to find next type
    int int_type = type;
    ++int_type;
    type = static_cast<Type>(int_type);
  }
  
  s << ends;
  return s.str();
}


static list<string> 
string_list(const string& s)
{
  list<string> l;
  l.push_back(s);
  return l;
}

static list<string> 
string_list(const string& s1, const string& s2)
{
  list<string> l;
  l.push_back(s1);
  l.push_back(s2);
  return l;
}

static list<string> 
string_list(const string& s1, const string& s2, const string& s3)
{
  list<string> l;
  l.push_back(s1);
  l.push_back(s2);
  l.push_back(s3);
  return l;
}

static list<string> 
string_list(const string& s1, const string& s2, const string& s3, const string& s4)
{
  list<string> l;
  l.push_back(s1);
  l.push_back(s2);
  l.push_back(s3);
  l.push_back(s4);
  return l;
}

END_NAMESPACE_STIR
