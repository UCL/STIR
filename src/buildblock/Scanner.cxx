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
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/Scanner.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/interfile_keyword_functions.h"
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
using std::cin;
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
    set_params(E931,string_list("ECAT 931"),  8, 192,2* 256, 510.0F, 13.5F,  3.129F,   
	       0,2,4,4,8,1); // 16 BUCKETS per ring in TWO rings - i.e. 32 buckets in total
  else if (scanner_type == E951)
    set_params(E951,string_list("ECAT 951"), 16, 192,2* 256, 510.0F, 6.75F,  3.129F,   
	       0,1,4,8,8, 1);
  else if (scanner_type == E953)
    set_params(E953,string_list("ECAT 953"), 16, 160,2* 192, 382.5F,  6.75F, 3.129F,   
	       static_cast<float>(15.*_PI/180),1,4,8,8, 1);
  else if (scanner_type == E921)
    set_params(E921,string_list("ECAT 921", "ECAT EXACT", "EXACT"), 24, 192,2* 192, 412.5F, 6.75F, 3.375F, 
	       static_cast<float>(15.*_PI/180),1,4,8,8, 1);
  else if (scanner_type == E925)
    set_params(E925,string_list("ECAT 925", "ECAT ART"), 24, 192,2* 192, 412.5F, 6.75F, 3.375F, 
	       static_cast<float>(15.*_PI/180),3,4,8,8, 1);
  else if (scanner_type == E961)
    set_params(E961,string_list("ECAT 961", "ECAT HR"), 24, 336,2* 392, 412.0F, 6.25F, 1.650F, 
	       static_cast<float>(13.*_PI/180),1,8,8,7, 1);
  else if (scanner_type == E962)
    set_params(E962,string_list("ECAT 962","ECAT HR+"), 32, 288,2* 288, 412.5F, 4.85F, 2.25F, 
	       0.F,4,3,8,8, 1);
  else if (scanner_type == E966)
    set_params(E966,string_list("ECAT EXACT 3D", "EXACT 3D", "ECAT HR++","ECAT 966"), 48, 288,2* 288, 412.5F, 4.850F, 2.250F, 
	       0,6,2,8,8, 1); 
  else if (scanner_type == RPT)
    set_params(RPT,string_list("PRT-1", "RPT"), 16, 128,2*192, 380,  6.75F, 3.1088F,   0,0,0,8,8, 1); 
  else if (scanner_type == RATPET)
    set_params(RATPET,string_list("RATPET"), 8, 56,2*56, 115/2.F,  6.25F, 1.65F,   
	       0,
	       1,16,8,7, 1); // HR block, 4 buckets per ring
  else if (scanner_type == Advance)
  {
    // 283 bins (non-uniform sampling) 
    // 281 bins (uniform sampling)
    /* crystal size 4x8x30*/
    set_params(Advance,string_list("GE Advance", "Advance"), 18, 283,281,2*336,471.875F, 8.5F, 1.970177F, 0,3,2,6,6, 1);
  }
  else if (scanner_type == DiscoveryLS)
  {
    // identical to Advance
    set_params(DiscoveryLS,string_list("GE Discovery LS", "Discovery LS"), 18, 283,281,2*336,471.875F, 8.5F, 1.970177F, 0,3,2,6,6, 1);
  }
  else if (scanner_type == DiscoveryST)
  {
    // 249 bins (non-uniform sampling) 
    // 223 bins (uniform sampling)
    /* crystal size: 6.3 x 6.3 x 30 mm*/
    set_params(DiscoveryST,string_list("GE Discovery ST", "Discovery ST"), 
	       24, 249,223,2*210,451.5F, 6.52916F, 3.1695F,
	       static_cast<float>(-4.54224*_PI/180),
	       4,2,6,6, 1);// TODO not sure about sign of view_offset
  }
  else if (scanner_type == HZLR)
    set_params(HZLR,string_list("Positron HZL/R"), 32, 256,2* 192, 780.0F, 5.1875F, 2.F, 0,0,0,0,0, 1);
  else if (scanner_type == HRRT)
    set_params(HRRT,string_list("HRRT"), 104, 288, 2*288, 234.765F, 
	       2.4375F, 1.21875F, 0,0,0,0,0, 2); // added by Dylan Togane
  else if (scanner_type == HiDAC)
    // all of these don't make any sense for the HiDAC
    set_params(HiDAC,string_list("HiDAC"), 0, 0, 0, 0.F, 0.F, 0.F, 0,0,0,0,0, 0);
  else if (scanner_type == User_defined_scanner)// zlong, 08-04-2004, Userdefined support
    set_params(User_defined_scanner, string_list("Userdefined"), 0, 0, 0, 0.F, 0.F, 0.F,0,0,0,0,0, 0);
  else
    { 
      // warning("Unknown scanner type used for initialisation of Scanner\n"); 
      set_params(Unknown_scanner,string_list("Unknown"), 0, 0, 0, 0.F, 0.F, 0.F, 0,0,0,0,0, 0);
    }
}


void
Scanner::
set_params(Type type_v,const list<string>& list_of_names_v,
	   int NoRings_v, 
	   int max_num_non_arccorrected_bins_v,
	   int num_detectors_per_ring_v,
	   float RingRadius_v,
	   float RingSpacing_v,
	   float BinSize_v, float intrTilt_v,
	   int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
	   int num_axial_crystals_per_block_v,int num_transaxial_crystals_per_block_v,
	   int num_detector_layers_v)
{
  set_params(type_v,list_of_names_v,NoRings_v,
	     max_num_non_arccorrected_bins_v,
	     max_num_non_arccorrected_bins_v,
	     num_detectors_per_ring_v, 
	     RingRadius_v, RingSpacing_v, 
	     BinSize_v, intrTilt_v,
	     num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
	     num_axial_crystals_per_block_v, num_transaxial_crystals_per_block_v,
	     num_detector_layers_v);
}

void
Scanner::
set_params(Type type_v,const list<string>& list_of_names_n,
	   int NoRings_v, 
	   int max_num_non_arccorrected_bins_v,
	   int default_num_arccorrected_bins_v,
	   int num_detectors_per_ring_v,
	   float RingRadius_v,
	   float RingSpacing_v,
	   float BinSize_v, float intrTilt_v,
	   int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
	   int num_axial_crystals_per_block_v,int num_transaxial_crystals_per_block_v,
	   int num_detector_layers_v)
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
  num_transaxial_blocks_per_bucket = num_transaxial_blocks_per_bucket_v;
  num_axial_blocks_per_bucket = num_axial_blocks_per_bucket_v;
  num_axial_crystals_per_block= num_axial_crystals_per_block_v;
  num_transaxial_crystals_per_block= num_transaxial_crystals_per_block_v;
  num_detector_layers = num_detector_layers_v;
}

Succeeded 
Scanner::
check_consistency() const
{
  if (intrinsic_tilt<-_PI || intrinsic_tilt>_PI)
    warning("Scanner %s: intrinsic_tilt is very large. maybe it's in degrees (but should be in radians)",
	    this->get_name().c_str());

  {
    if (get_num_transaxial_crystals_per_block() <= 0 ||
	get_num_transaxial_blocks() <= 0)
      warning("Scanner %s: transaxial block info is not set",
	      this->get_name().c_str());
    else
      {
	const int dets_per_ring =
	  get_num_transaxial_blocks() *
	  get_num_transaxial_crystals_per_block();
	if ( dets_per_ring != get_num_detectors_per_ring())
	  { 
	    warning("Scanner %s: inconsistent transaxial block info",
		    this->get_name().c_str()); 
	    return Succeeded::no; 
	  }
      }
  }
  {
    if (get_num_transaxial_blocks_per_bucket() <= 0 ||
	get_num_transaxial_buckets() <=0)
      warning("Scanner %s: transaxial bucket info is not set",
	      this->get_name().c_str());
    else
      {
	const int blocks_per_ring =
	  get_num_transaxial_buckets() *
	  get_num_transaxial_blocks_per_bucket();
	if ( blocks_per_ring != get_num_transaxial_blocks())
	  { 
	    warning("Scanner %s: inconsistent transaxial block/bucket info",
		    this->get_name().c_str()); 
	    return Succeeded::no; 
	  }
      }
  }
  {
    if (get_num_axial_crystals_per_block() <= 0 ||
	get_num_axial_blocks() <=0)
      warning("Scanner %s: axial block info is not set",
	      this->get_name().c_str());
    else
      {
	const int dets_axial =
	  get_num_axial_blocks() *
	  get_num_axial_crystals_per_block();
	if ( dets_axial != get_num_rings())
	  { 
	    warning("Scanner %s: inconsistent axial block info",
		    this->get_name().c_str()); 
	    return Succeeded::no; 
	  }
      }
  }
  {
    if (get_num_axial_blocks_per_bucket() <= 0 ||
	get_num_axial_buckets() <=0)
      warning("Scanner %s: axial bucket info is not set",
	      this->get_name().c_str());
    else
      {
	const int blocks_axial =
	  get_num_axial_buckets() *
	  get_num_axial_blocks_per_bucket();
	if ( blocks_axial != get_num_axial_blocks())
	  { 
	    warning("Scanner %s: inconsistent axial block/bucket info",
		    this->get_name().c_str()); 
	    return Succeeded::no; 
	  }
      }
  }

  return Succeeded::yes;
}

#if 0
Scanner::Scanner(Type type_v,
		 const list<string>& list_of_names_v,
		 int num_detectors_per_ring_v,
		 int NoRings_v,	
		 int max_num_non_arccorrected_bins, float RingRadius_v, 
		 float RingSpacing_v, float BinSize_v, float intrTilt_v,
		 int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
		 int num_axial_crystals_per_block_v,int num_transaxial_crystals_per_block_v,
		 int num_detector_layers_v)
{
  set_params(type_v,list_of_names_v,NoRings_v,
	     max_num_non_arccorrected_bins,
	     num_detectors_per_ring_v, 
	     RingRadius_v, RingSpacing_v, 
	     BinSize_v, intrTilt_v,
	     num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
	     num_axial_crystals_per_block_v, num_transaxial_crystals_per_block_v,
	     num_detector_layers_v);
}

Scanner::Scanner(Type type_v,const string& name,
		 int num_detectors_per_ring_v, int NoRings_v, int max_num_non_arccorrected_bins, 
		 float RingRadius_v, float RingSpacing_v, 
		 float BinSize_v, float intrTilt_v,
		 int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
		 int num_axial_crystals_per_block_v,int num_transaxial_crystals_per_block_v,
		 int num_detector_layers_v)
{
  set_params(type_v,string_list(name),NoRings_v,
	     max_num_non_arccorrected_bins,
	     num_detectors_per_ring_v, 
	     RingRadius_v, RingSpacing_v, 
	     BinSize_v, intrTilt_v,
	     num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
	     num_axial_crystals_per_block_v,transaxial_crystals_per_block_v,
	     num_detector_layers_v);

}
#endif

Scanner::Scanner(Type type_v,
		 const list<string>& list_of_names_v,
		 int num_detectors_per_ring_v,
		 int NoRings_v,	
		 int max_num_non_arccorrected_bins,
		 int default_num_arccorrected_bins,
		 float RingRadius_v, 
		 float RingSpacing_v, float BinSize_v, float intrTilt_v,
		 int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
		 int num_axial_crystals_per_block_v,int num_transaxial_crystals_per_block_v,
		 int num_detector_layers_v)
{
  set_params(type_v,list_of_names_v,NoRings_v,
	     max_num_non_arccorrected_bins,
	     default_num_arccorrected_bins,
	     num_detectors_per_ring_v, 
	     RingRadius_v, RingSpacing_v, 
	     BinSize_v, intrTilt_v,
	     num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
	     num_axial_crystals_per_block_v, num_transaxial_crystals_per_block_v,
	     num_detector_layers_v);
	
}

Scanner::
Scanner(Type type_v,const string& name,
	int num_detectors_per_ring_v, int NoRings_v, 
	int max_num_non_arccorrected_bins,
	int default_num_arccorrected_bins,
	float RingRadius_v, float RingSpacing_v, 
	float BinSize_v, float intrTilt_v,
	int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
	int num_axial_crystals_per_block_v,int num_transaxial_crystals_per_block_v,
	int num_detector_layers_v)
{
  set_params(type_v,string_list(name),NoRings_v,
	     max_num_non_arccorrected_bins,
	     default_num_arccorrected_bins,
	     num_detectors_per_ring_v, 
	     RingRadius_v, RingSpacing_v, 
	     BinSize_v, intrTilt_v,
	     num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
	     num_axial_crystals_per_block_v,num_transaxial_crystals_per_block_v,
	     num_detector_layers_v);

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
  s<<"Scanner parameters:= "<<'\n';

  s << "Scanner type := " << get_name() <<'\n';     

  s << "Number of Rings                        := " << num_rings << '\n';
  s << "Number of detectors per ring           := " << get_num_detectors_per_ring() << '\n';
  s << "ring diameter (cm)                     := " << get_ring_radius()*2./10 << '\n'
    << "distance between rings (cm)            := " << get_ring_spacing()/10 << '\n'
    << "bin size (cm)                          := " << get_default_bin_size()/10. << '\n'
    << "view offset (degrees)                  := " << get_default_intrinsic_tilt()*180/_PI << '\n';
  // block/bucket description
  s << "Number of blocks per bucket in transaxial direction  := "
    << get_num_transaxial_blocks_per_bucket() << '\n'
    << "Number of blocks per bucket in axial direction       := "
    << get_num_axial_blocks_per_bucket() << '\n'
    << "Number of crystals per block in axial direction      := "
    << get_num_axial_crystals_per_block() << '\n'
    << "Number of crystals per block in transaxial direction := "
    << get_num_transaxial_crystals_per_block() << '\n'
    << "Number of detector layers                            := "
    << get_num_detector_layers() << '\n';
  s << "Maximum number of nonarccorrected bins := "
     << get_max_num_non_arccorrected_bins() << '\n'
    << "Default number of arccorrected bins    := "
    << get_default_num_arccorrected_bins() << '\n';
  s<<"end Scanner parameters:= "<<'\n';

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

  return s.str();
}

/************************************************
 static members
 *************************************************/
Scanner* Scanner::ask_parameters() 
{

  cerr << list_all_names();

  const string name=ask_string("Enter the name of the scanner");

  //get the type from the name itself
  Scanner* scanner_ptr = 
    get_scanner_from_name(name);

  if (scanner_ptr->type != Unknown_scanner && scanner_ptr->type != User_defined_scanner)
    return scanner_ptr;

  if (scanner_ptr->type == Unknown_scanner)
    cerr << "I didn't recognise the scanner you entered.";
  cerr << "I'll ask lots of questions\n";
  
  while (true)
    {
      int num_detectors_per_ring = 
	ask_num("Enter number of detectors per ring:",0,2000,128);
  
      int NoRings = 
	ask_num("Enter number of rings :",0,128,16);
  
      int NoBins = 
	ask_num("Enter number of bins: ",0,2000,128);
  
      float RingRadius=
	ask_num("Enter ring radius (in mm): ",0.F,600.F,256.F);
  
      float RingSpacing= 
	ask_num("Enter ring spacing (in mm): ",0.F,20.F,6.75F);
  
      float BinSize= 
	ask_num("Enter bin size (in mm):",0.F,20.F,3.75F);
      float intrTilt=
	ask_num("Enter intrinsic_tilt (in degrees):",-180.F,360.F,0.F);
      int TransBlocksPerBucket = 
	ask_num("Enter number of transaxial blocks per bucket: ",0,10,2);
      int AxialBlocksPerBucket = 
	ask_num("Enter number of axial blocks per bucket: ",0,10,6);
      int AxialCrystalsPerBlock = 
	ask_num("Enter number of axial crystals per block: ",0,12,8);
      int TransaxialCrystalsPerBlock = 
	ask_num("Enter number of transaxial crystals per block: ",0,12,8);
      int num_detector_layers =
	ask_num("Enter number of layers per block: ",1,100,1);
      Type type = User_defined_scanner;
  
      Scanner* scanner_ptr =
	new Scanner(type,string_list(name),
		    num_detectors_per_ring,  NoRings, 
		    NoBins, NoBins, 
		    RingRadius, RingSpacing, 
		    BinSize,intrTilt*float(_PI)/180,
		    AxialBlocksPerBucket,TransBlocksPerBucket,
		    AxialCrystalsPerBlock,TransaxialCrystalsPerBlock,
		    num_detector_layers );
  
      if (scanner_ptr->check_consistency()==Succeeded::yes ||
	  !ask("Ask questions again?",true))
	return scanner_ptr;
  
      delete scanner_ptr;
    } // infinite loop
}


Scanner *
Scanner::get_scanner_from_name(const string& name)
{ 
  Scanner * scanner_ptr;

  const string matching_name =
    standardise_interfile_keyword(name);
  Type type= E931; 
  while (type != Unknown_scanner)
  {
    scanner_ptr = new Scanner(type);
    const list<string>& list_of_names = scanner_ptr->get_all_names();
    for (std::list<string>::const_iterator iter =list_of_names.begin();
	 iter!=list_of_names.end();
	   ++iter)
      {
	const string matching_scanner_name =
	  standardise_interfile_keyword(*iter);
	if (matching_scanner_name==matching_name)
	  return scanner_ptr;
      }
    
    // we didn't find it yet
    delete scanner_ptr;
    // tricky business to find next type
    int int_type = type;
    ++int_type;
    type = static_cast<Type>(int_type);
  }
  // it's not in the list
  return new Scanner(Unknown_scanner);
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
  while (type != Unknown_scanner)
  {
    scanner_ptr = new Scanner(type);
    s << scanner_ptr->list_names() << '\n';
    
    delete scanner_ptr;
    // tricky business to find next type
    int int_type = type;
    ++int_type;
    type = static_cast<Type>(int_type);
  }
  
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
