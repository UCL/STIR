//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class Scanner

  \author Claire Labbe
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SCANNER_H__
#define __stir_SCANNER_H__

#include "stir/common.h"
#include "stir/DetectionPosition.h"
#include <string>
#include <list>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::list;
#endif

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup buildblock
  \brief A class for storing some info on the scanner

  Currently really only appropriate for cylindrical PET scanners
  \todo  
    a hierarchy distinguishing between different types of scanners
  \todo derive from ParsingObject
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

  // E931 HAS to be first, Unknown_scanner HAS to be last
  // also, the list HAS to be consecutive (so DO NOT assign numbers here)
  // finally, test_Scanner assumes E966 is the last in the list of CTI scanners
  // supported by ecat_model from the LLN matrix library
  // 08-3-2004, zlong, add user defined scanner
  //! enum for all predefined scanners
  /* \a Userdefined_scanner can be used to set arbitrary scanner geometry info.
     \a Unknown_scanner will be used when parsing (e.g. from an Interfile header) 
     to flag up an error and do some guess work in trying to recognise the scanner from 
     any given parameters.
  */
  enum Type {E931,E951,E953,E921,E925,E961,E962,E966,RPT,HiDAC,
	     Advance, DiscoveryLS, DiscoveryST,
	     HZLR, RATPET,HRRT,
	     User_defined_scanner,
	     Unknown_scanner};
  
  
  //! constructor that takes scanner type as an input argument
  Scanner(Type scanner_type);
  //! constructor -(list of names)
  /*! size info is in mm
      \param intrinsic_tilt value in radians, \see get_default_intrinsic_tilt()
      \warning calls error() when block/bucket info are inconsistent
   */
  Scanner(Type type,const list<string>& list_of_names,
	  int num_detectors_per_ring, int NoRings, 
	  int max_num_non_arccorrected_bins,
	  int default_num_arccorrected_bins,
	  float RingRadius, float RingSpacing, 
	  float BinSize, float intrinsic_tilt,
	  int num_axial_blocks_per_bucket, int num_transaxial_blocks_per_bucket,
	  int num_axial_crystals_per_block,int num_transaxial_crystals_per_block,
	  int num_detector_layers);

  //! constructor ( a single name)
  /*! size info is in mm
      \param intrinsic_tilt value in radians, \see get_default_intrinsic_tilt()
      \warning calls error() when block/bucket info are inconsistent
   */
  Scanner(Type type_v,const string& name,
	  int num_detectors_per_ring, int NoRings_v, 
	  int max_num_non_arccorrected_bins,
	  int default_num_arccorrected_bins,
	  float RingRadius_v, float RingSpacing_v, 
	  float BinSize_v, float intrinsic_tilt,
	  int num_axial_blocks_per_bucket, int num_transaxial_blocks_per_bucket,
	  int num_axial_crystals_per_block,int num_transaxial_crystals_per_block,
	  int num_detector_layers);
#if 0
  //! constructor with list of names, putting max_num_non_arccorrected bins and default_num_arccorrected_bins equal
  /*! size info is in mm
      \param intrinsic_tilt value in radians, \see get_default_intrinsic_tilt()
      \warning calls error() when block/bucket info are inconsistent
   */
  Scanner(Type type_v,const list<string>& list_of_names,
         int num_detectors_per_ring, int NoRings_v, 
	 int max_num_non_arccorrected_bins,
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v,
	  int num_axial_blocks_per_bucket, int num_transaxial_blocks_per_bucket,
	  int num_axial_crystals_per_block,int num_transaxial_crystals_per_block,
	  int num_detector_layers);

 //! constructor - one name given and max_num_non_arccorrected bins only
  Scanner(Type type_v,const string names,
         int num_detectors_per_ring, int NoRings_v, 
	 int max_num_non_arccorrected_bins, 
	 float RingRadius_v, float RingSpacing_v, 
         float BinSize_v, float intrTilt_v,
	 int num_axial_blocks_per_bucket=0, int num_transaxial_blocks_per_bucket=0,
         int num_axial_crystals_per_block=0,int num_transaxial_crystals_per_block=0);
#endif

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
  //! checks consistency 
  /*! Calls warning() with diagnostics when there are problems
   */
  Succeeded check_consistency() const;

  //! \name Functions returning geometrical info
  //@{

  //! get number of rings
  inline int get_num_rings() const;
  //! get the number of detectors per ring
  inline int get_num_detectors_per_ring() const;
  //! get the  maximum number of arccorrected tangential positions
  /*! \warning name is not in standard STIR terminology. Should be
      \c get_max_num_non_arccorrected_tangential_poss() or so.
      Even better would be to replace it by a name relating
      to the number of detectors that are in coincidence with a
      a single detector in the ring, as that is the physical
      reason why there is a maximum number anyway.
      \todo change name
  */
  inline int get_max_num_non_arccorrected_bins() const;
  //! get the default number of arccorrected tangential positions
  /*! \warning name is not in standard STIR terminology. Should be
      \c get_max_default_num_arccorrected_tangential_poss() or so.
      \todo change name, mabe refering to the fan of detectors 
      in coincidence or so
  */
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
  //! get intrinsic tilt in raw sinogram data (in radians)
  /*! Some scanners construct sinograms where the first view does not
      correspond to the vertical. This angle tells you how much the
      image will be rotated when this tilt is ignored in the reconstruction
      algorithm. It uses the same coordinate system as ProjDataInfo::get_phi().

      \todo we still have to decide if ProjDataInfo::get_phi() will take 
      this tilt into account or not. At present, STIR ignores the intrinsic tilt.
  */
  inline float get_default_intrinsic_tilt() const;
  //! \name Info on crystals per block etc.
  /*! At present, these functions follow CTI terminology, but the concepts
      are similar for other scanners.
      \li \c crystal the smallest detection unit
      \li \c block several crystals are grouped in a block, this can be in
          3 dimensions (see layer). This information could be useful for finding the
	  geometry of the scanner, but we would plenty more
	  size info presumably.
      \li \c layer Some scanners have multiple layers of detectors to give
          Depth Of Interaction information
      \li \c bucket several \c blocks send detected events to one \c bucket.
          This has consequences for the dead-time modelling. For instance,
	  one bucket could have much more singles than another, and hence
	  presumably higher singles-dead-time.

      \warning Only sensible for discrete detector-based scanners.
      \todo Most scanners do not have this info filled in at present.
  */      
  //@{
  //! get number of transaxial blocks per bucket
  inline int get_num_transaxial_blocks_per_bucket() const;
  //! get number of axial blocks per bucket
  inline int get_num_axial_blocks_per_bucket() const;	
  //! get number of crystals in the axial direction
  inline int get_num_axial_crystals_per_block() const;	
  //! get number of transaxial crystals 
  inline int get_num_transaxial_crystals_per_block() const;
  //! get number of crystal layers (for DOI)
  inline int get_num_detector_layers() const;	
  //! get number of axial blocks
  inline int get_num_axial_blocks() const;	
  //! get number of axial blocks
  inline int get_num_transaxial_blocks() const;	
  //! get number of axial buckets
  inline int get_num_axial_buckets() const;	
  //! get number of axial buckets
  inline int get_num_transaxial_buckets() const;	

  //! get number of axial crystals per singles unit
  inline int get_num_axial_crystals_per_singles_unit() const;
  //! get number of transaxial crystals per singles unit.
  inline int get_num_transaxial_crystals_per_singles_unit() const;
  // TODO accomodate more complex geometries of singles units.
  /*  inline int get_num_crystal_layers_per_singles_unit() const; */
  //! get number of axial singles units
  inline int get_num_axial_singles_units() const;
  //! get number of transaxial singles unit
  inline int get_num_transaxial_singles_units() const;
  /* inline int get_num_layers_singles_units() const; */
  inline int get_num_singles_units() const;


  //@} (end of block/bucket info)

  //@} (end of get geometrical info)

  //! \name Functions setting info
  /*! Be careful to keep consistency by setting all relevant parameters*/
  //@{
  // zlong, 08-04-2004, add set_methods
  //! set scanner type
  inline void set_type(const Type & new_type);
  //! set number of rings
  inline void set_num_rings(const int & new_num);
  //! set the namber of detectors per ring
  inline void set_num_detectors_per_ring(const int & new_num) ;
  //! set the  maximum number of arccorrected bins
  inline void set_max_num_non_arccorrected_bins(const int & new_num) ;
  //! set the default number of arccorrected_bins
  inline void set_default_num_arccorrected_bins(const int & new_num) ;
 //! set ring radius
  inline void set_ring_radius(const float & new_radius);
 //! set ring spacing 
  inline void set_ring_spacing(const float & new_spacing);
  //! set default arc-corrected bin size
  inline void set_default_bin_size(const float &new_size);
  //! in degrees
  inline void set_default_intrinsic_tilt(const float & new_tilt);
  //! \name Info on crystals per block etc.
  //@{
  //! set number of transaxial blocks per bucket
  inline void set_num_transaxial_blocks_per_bucket(const int & new_num);
  //! set number of axial blocks per bucket
  inline void set_num_axial_blocks_per_bucket(const int & new_num);	
  //! set number of crystals in the axial direction
  inline void set_num_axial_crystals_per_block(const int & new_num);	
  //! set number of transaxial crystals 
  inline void set_num_transaxial_crystals_per_block(const int & new_num);
  //! set number of crystal layers (for DOI)
  inline int set_num_detector_layers(const int& num_num);	
  //! set number of axial crystals per singles unit
  inline void set_num_axial_crystals_per_singles_unit(const int & new_num);	
  //! set number of transaxial crystals per singles unit
  inline void set_num_transaxial_crystals_per_singles_unit(const int & new_num);
  // TODO accomodate more complex geometries of singles units.

  //@} (end of block/bucket info)

  //@} (end of set info)
  
  // Calculate a singles bin index from axial and transaxial singles bin coordinates.
  inline int get_singles_bin_index(int axial_index, int transaxial_index) const;

  // Method used to calculate a singles bin index from
  // a detection position.
  inline int get_singles_bin_index(const DetectionPosition<>& det_pos) const; 
 

  // Get the axial singles bin coordinate from a singles bin.
  inline int get_axial_singles_unit(int singles_bin_index) const;

  // Get the transaxial singles bin coordinate from a singles bin.
  inline int get_transaxial_singles_unit(int singles_bin_index) const;
  

private:
  Type type;
  list<string> list_of_names;
  int num_rings;		/* number of direct planes */
  int max_num_non_arccorrected_bins; 
  int default_num_arccorrected_bins; /* default number of bins */
  int num_detectors_per_ring;	
  float ring_radius;	/*! detector radius in mm*/
  float ring_spacing;	/*! ring separation in mm*/
  float bin_size;		/*! arc-corrected bin size in mm (spacing of transaxial elements) */
  float intrinsic_tilt;		/*! intrinsic tilt in radians*/

  int num_transaxial_blocks_per_bucket;	/* transaxial blocks per bucket */
  int num_axial_blocks_per_bucket;	/* axial blocks per bucket */
  int num_axial_crystals_per_block;	/* number of crystals in the axial direction */
  int num_transaxial_crystals_per_block;/* number of transaxial crystals */
  int num_detector_layers;

  int num_axial_crystals_per_singles_unit;
  int num_transaxial_crystals_per_singles_unit;


  // ! set all parameters, case where default_num_arccorrected_bins==max_num_non_arccorrected_bins
  void set_params(Type type_v,const list<string>& list_of_names,
                  int NoRings_v, 
		  int max_num_non_arccorrected_bins,
		  int num_detectors_per_ring,
		  float RingRadius_v,
		  float RingSpacing_v,
		  float BinSize_v, float intrTilt_v,
		  int num_axial_blocks_per_bucket, int num_transaxial_blocks_per_bucket, 
		  int num_axial_crystals_per_block,int num_transaxial_crystals_per_block,
		  int num_detector_layers);

  // ! set all parameters
  void set_params(Type type_v,const list<string>& list_of_names,
                  int NoRings_v, 
		  int max_num_non_arccorrected_bins,
		  int default_num_arccorrected_bins,
		  int num_detectors_per_ring,
		  float RingRadius_v,
		  float RingSpacing_v,
		  float BinSize_v, float intrTilt_v,
		  int num_axial_blocks_per_bucket, int num_transaxial_blocks_per_bucket,
		  int num_axial_crystals_per_block,int num_transaxial_crystals_per_block,
		  int num_detector_layers);

};

END_NAMESPACE_STIR

#include "stir/Scanner.inl"

#endif
 
