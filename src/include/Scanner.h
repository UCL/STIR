//
// $Id$: $Date$
//
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class Scanner

  \author Claire Labbe
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  \date $Date$
  \version $Revision$

*/

#ifndef __SCANNER_H__
#define __SCANNER_H__

#include "Tomography_common.h"
#include <string>
#include <list>

#ifndef TOMO_NO_NAMESPACES
using std::string;
using std::list;
#endif

START_NAMESPACE_TOMO


/*!
  \ingroup buildblock
  \brief A class for storing some info on the scanner

    TODO: 
    a hierarchy distinguishing between different types of scanners
*/
class Scanner 
{
 public:

   /************* static members*****************************/
  static Scanner * ask_parameters();

  //! get the scanner pointer from the name
  static Scanner * get_scanner_from_name(const string& name);
  //! get the list of all names for the particular scanner
  static string list_all_names();

  // E931 HAS to be first, Unknown_Scanner HAS to be last
  // also, the list HAS to be consecutive (so DO NOT assign numbers here)
  enum Type {E931,E951,E953,E921,E925,E961,E962,E966,ART,RPT,HiDAC,Advance, HZLR, Unknown_Scanner};
  

  //! constructor that takes scanner type as an input argument
  Scanner(Type scanner_type);
  //! constructor-max_num_non_arccorrected bins and default_num_arccorrected_bins(list of names)
  Scanner(Type type_v,const list<string>& list_of_names,
         int num_detectors_per_ring, int NoRings_v, 
	 int max_num_non_arccorrected_bins,
	 int default_num_arccorrected_bins,
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v);
 //! construcor - max_num_non_arccorrected bins and default_num_arccorrected_bins ( a single name)
  Scanner(Type type_v,const string names,
         int num_detectors_per_ring, int NoRings_v, 
	 int max_num_non_arccorrected_bins,
	 int default_num_arccorrected_bins,
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v);

  //! constructor with list of names and max_num_non_arccorrected bins only
  Scanner(Type type_v,const list<string>& list_of_names,
         int num_detectors_per_ring, int NoRings_v, 
	 int max_num_non_arccorrected_bins,
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v);
 //! constructor - one name given and max_num_non_arccorrected bins only
  Scanner(Type type_v,const string names,
         int num_detectors_per_ring, int NoRings_v, 
	 int max_num_non_arccorrected_bins, 
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v);

  //! get scanner parameters as a string
  string parameter_info() const;
  //! get the scanner name
  const string& get_name() const;
  //! get all scanner names as a list of strings
  const list<string>& get_all_names() const;
  //! get all scanner names as a string
  string list_names() const;

  //! comparison operator
  bool operator ==(const Scanner& scanner) const;
  inline bool operator !=(const Scanner& scanner) const;

  //! get scanner type
  inline Type get_type() const;

  //! get number of rings
  inline int get_num_rings() const;
  //! get the namber of detectors per ring
  inline int get_num_detectors_per_ring() const;
  //! get the  maximum number of arccorrected bins
  inline int get_max_num_non_arccorrected_bins() const;
  //! get the default number of arccorrected_bins
  inline int get_default_num_arccorrected_bins() const;
  //! get maximum number of views
  /*! This is simply get_num_detectors_per_ring()/2 */
  inline int get_max_num_views() const;
 //! get ring radius
  inline float get_ring_radius() const;
 //! get ring spacing 
  inline float get_ring_spacing() const;
  //! get default arc-corrected bin size
  inline float get_default_bin_size() const;
  //! in degrees
  inline float get_default_intrinsic_tilt() const;


private:
  Type type;  		/* model number as an ascii string */
  list<string> list_of_names;
  int num_rings;		/* number of direct planes */
  int max_num_non_arccorrected_bins; 
  int default_num_arccorrected_bins; /* default number of bins */
  int num_detectors_per_ring;	
  float ring_radius;	/* detector radius in mm*/
  float ring_spacing;	/* plane separation in mm*/
  float bin_size;		/* arc-corrected bin size in mm (spacing of transaxial elements) */
  float intrinsic_tilt;		/* intrinsic tilt in degrees*/

  // ! set all parameters, case where default_num_arccorrected_bins==max_num_non_arccorrected_bins
  void set_params(Type type_v,const list<string>& list_of_names,
                  int NoRings_v, 
		  int max_num_non_arccorrected_bins,
		  int num_detectors_per_ring,
		  float RingRadius_v,
		  float RingSpacing_v,
		  float BinSize_v, float intrTilt_v);

  // ! set all parameters
  void set_params(Type type_v,const list<string>& list_of_names,
                  int NoRings_v, 
		  int max_num_non_arccorrected_bins,
		  int default_num_arccorrected_bins,
		  int num_detectors_per_ring,
		  float RingRadius_v,
		  float RingSpacing_v,
		  float BinSize_v, float intrTilt_v);

};

END_NAMESPACE_TOMO

#include"Scanner.inl"

#endif
 
