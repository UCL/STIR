/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2010, Hammersmith Imanet Ltd
    Copyright (C) 2011-2013, King's College London
    Copyright (C) 2016, University of Hull
    Copyright (C) 2016, 2019, 2021, 2023 UCL
    Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C 2017-2018, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::Scanner

  \author Nikos Efthimiou
  \author Claire Labbe
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Charalampos Tsoumpas
  \author Ottavia Bertolli
  \author Palak Wadhwa
  \author PARAPET project
  \author Parisa Khateri

*/
#ifndef __stir_buildblock_SCANNER_H__
#define __stir_buildblock_SCANNER_H__

#include "stir/DetectionPosition.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/DetectorCoordinateMap.h"
#include "stir/shared_ptr.h"
#include <string>
#include <list>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>

START_NAMESPACE_STIR

class Succeeded;
/*!
  \ingroup buildblock
  \brief A class for storing some info on the scanner

  This class stores geometrical info etc on the scanner.
  \warning Currently really only appropriate for cylindrical PET scanners

  \par information on blocks, buckets etc
  This class gives some informations on crystals, blocks etc. However, this
  is currently (i.e. at least up to STIR 2.1) used in very few places.
  For ECAT scanners, this information is used to read the normalisation .n 
  files and computed dead-time correction etc. For all other scanners, STIR
  currently ignores this info. This might change in the future of course.
  
  At present, these functions follow CTI terminology, but the concepts
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
      \li \c singles_unit (non-standard terminology)
          Most scanners report the singles detected during the acquisition.
          Some scanners (such as GE scanners) report singles for every crystal,
          while others (such as CTI scanners) give only singles for a 
          collection of blocks. A \c singles_unit is then a set of crystals
          for which we can get singles rates.

      A further complication is that some scanners (including many Siemens scanners) 
      insert virtual crystals in the sinogram data (corresponding to gaps between
      detector blocks). We currently define the blocks as the "virtual" ones,
      but provide extra members to find out how many of these virtual crystals there are.

      \warning This information is only sensible for discrete detector-based scanners.
      \warning Currently, in a TOF compatible scanner template, the last three types have to
                be explicitly defined to avoid ambiguity.
      \warning The energy resolution has to be specified but it is used only for scatter correction.
      \warning In order to define a nonTOF scanner the timing resolution has to be set to 0 or 1.
                Anything else will trigger a TOF reconstruction.
      \todo Some scanners do not have all info filled in at present. Values are then
      set to 0.

      \warning You have to call set_up() after using the \c set_* functions (except set_params()).

  \todo  
    a hierarchy distinguishing between different types of scanners
  \todo derive from ParsingObject
*/
class Scanner 
{
    friend class BlocksTests;
    
 public:

   /************* static members*****************************/
  static Scanner * ask_parameters();

  //! get the scanner pointer from the name
  static Scanner * get_scanner_from_name(const std::string& name);
  //! get a string listing names for all predefined scanners
  /* \return a string with one line per predefined scanner, listing the predefined names for
     that scanner (separated by a comma)
  */
  static std::string list_all_names();
  //! get a list with the names for each predefined scanner
  /* \return a list of strings, each element is a name of a predefined scanner.
     If a scanner can have multiple names, only one name is returned, i.e.
     the list has the same length as the number of predefined scanners.
  */
  static std::list<std::string> get_names_of_predefined_scanners();

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
  enum Type {E931, E951, E953, E921, E925, E961, E962, E966, E1080, test_scanner, Siemens_mMR,Siemens_mCT, Siemens_Vision_600, RPT,HiDAC,
	     Advance, DiscoveryLS, DiscoveryST, DiscoverySTE, DiscoveryRX, Discovery600, PETMR_Signa,
	     Discovery690, DiscoveryMI3ring, DiscoveryMI4ring, DiscoveryMI5ring,
	     HZLR, RATPET, PANDA, HYPERimage, nanoPET, HRRT, Allegro, GeminiTF, SAFIRDualRingPrototype,             UPENN_5rings, UPENN_5rings_no_gaps, UPENN_6rings, UPENN_6rings_no_gaps,
             User_defined_scanner,
	     Unknown_scanner};

  virtual ~Scanner() {}

  //! constructor that takes scanner type as an input argument
  Scanner(Type scanner_type);


  //! constructor -(list of names)
  /*! size info is in mm
      \param intrinsic_tilt_v value in radians, \see get_intrinsic_azimuthal_tilt()
      \param scanner_geometry_v \see set_scanner_geometry()
      \warning calls error() when block/bucket info are inconsistent
   */
  Scanner(Type type_v, const std::list<std::string>& list_of_names_v,
          int num_detectors_per_ring_v, int num_rings_v,
          int max_num_non_arccorrected_bins_v,
          int default_num_arccorrected_bins_v,
          float inner_ring_radius_v, float average_depth_of_interaction_v,
          float ring_spacing_v, float bin_size_v, float intrinsic_tilt_v,
          int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
          int num_axial_crystals_per_block_v, int num_transaxial_crystals_per_block_v,
          int num_axial_crystals_per_singles_unit_v,
          int num_transaxial_crystals_per_singles_unit_v,
          int num_detector_layers_v,
          float energy_resolution_v = -1.0f,
          float reference_energy_v = -1.0f,
          short int max_num_of_timing_poss = -1,
          float size_timing_pos = -1.0f,
          float timing_resolution = -1.0f,
          const std::string& scanner_geometry_v = "Cylindrical",
          float axial_crystal_spacing_v = -1.0f,
          float transaxial_crystal_spacing_v = -1.0f,
          float axial_block_spacing_v = -1.0f,
          float transaxial_block_spacing_v = -1.0f,
          const std::string& crystal_map_file_name = "");            

  //! constructor ( a single name)
  /*! size info is in mm
      \param intrinsic_tilt value in radians, \see get_intrinsic_azimuthal_tilt()
      \param scanner_geometry_v \see set_scanner_geometry()
      \warning calls error() when block/bucket info are inconsistent
   */
  Scanner(Type type_v, const std::string& name,
          int num_detectors_per_ring_v, int num_rings_v,
          int max_num_non_arccorrected_bins_v,
          int default_num_arccorrected_bins_v,
          float inner_ring_radius_v, float average_depth_of_interaction_v,
          float ring_spacing_v, float bin_size_v, float intrinsic_tilt_v,
          int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
          int num_axial_crystals_per_block_v, int num_transaxial_crystals_per_block_v,
          int num_axial_crystals_per_singles_unit_v,
          int num_transaxial_crystals_per_singles_unit_v,
          int num_detector_layers_v,
          float energy_resolution_v = -1.0f,
          float reference_energy_v = -1.0f,
          short int max_num_of_timing_poss = -1,
          float size_timing_pos = -1.0f,
          float timing_resolution = -1.0f,
          const std::string& scanner_geometry_v = "Cylindrical",
          float axial_crystal_spacing_v = -1.0f,
          float transaxial_crystal_spacing_v = -1.0f,
          float axial_block_spacing_v = -1.0f,
          float transaxial_block_spacing_v = -1.0f,
          const std::string& crystal_map_file_name = "");

  //! Initialise internal geometry
  /*! Currently called in the set_params() functions, but needs to be
      called explicitly when afterwards using any of the other \c set_ functions
  */
  virtual void set_up();

  //! get scanner parameters as a std::string
  std::string parameter_info() const;
  //! get the scanner name
  const std::string& get_name() const;
  //! get all scanner names as a list of strings
  const std::list<std::string>& get_all_names() const;
  //! get all scanner names as a string
  std::string list_names() const;

  //! comparison operator
  bool operator ==(const Scanner& scanner) const;
  inline bool operator !=(const Scanner& scanner) const;

  //! get scanner type
  inline Type get_type() const;
  //! checks consistency 
  /*! Calls warning() with diagnostics when there are problems
   * N.E: Should something check be added for TOF information?
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
  //! get inner ring radius
  inline float get_inner_ring_radius() const;
  //! get maximum field of view radius
  inline float get_max_FOV_radius() const;
  //! get effective ring radius
  inline float get_effective_ring_radius() const;
  //! get average depth of interaction
  inline float get_average_depth_of_interaction() const;
  //! get ring spacing 
  inline float get_ring_spacing() const;
  //! get default arc-corrected bin size
  inline float get_default_bin_size() const;
  //! get intrinsic tilt in raw sinogram data (in radians)
  /*! Some scanners construct sinograms where the first view does not
      correspond to the vertical. This angle tells you how much the
      image will be rotated when this tilt is ignored in the reconstruction
      algorithm. It uses the same coordinate system as ProjDataInfo::get_phi().
  */
  inline float get_intrinsic_azimuthal_tilt() const;
  //! \name Info on crystals per block etc.
  //@{
  //! get number of transaxial blocks per bucket
  inline int get_num_transaxial_blocks_per_bucket() const;
  //! get number of axial blocks per bucket
  inline int get_num_axial_blocks_per_bucket() const;   
  //! get number of crystals in the axial direction
  inline int get_num_axial_crystals_per_block() const;  
  //! get number of transaxial crystals 
  inline int get_num_transaxial_crystals_per_block() const;
  //! get crystals in a bucket
  inline int get_num_transaxial_crystals_per_bucket() const;
  //! get crystals in a bucket
  inline int get_num_axial_crystals_per_bucket() const;
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
  //! Get the maximum number of TOF bins.     
  /*! \return will be 0 or negative if not known */
  inline int get_max_num_timing_poss() const;
  //! Get the size for one (unmashed) TOF bin in picoseconds
  /*!
    \return will be 0 or negative if not known
    \todo change name to \c get_size_of_timing_pos_in_ps
  */
  inline float get_size_of_timing_pos() const;
  //! Get the timing resolution of the scanner in picoseconds
  /*!
    \return will be 0 or negative if not known
    \todo change name to \c get_size_of_timing_pos_in_ps
    */
  inline float get_timing_resolution() const;
  //! Get the full width of the coincidence window in picoseconds
  /*!
    This is often written as \f$2\tau\f$ and is usually around 4000ps.
    It is determined from \c get_max_num_timing_poss() and \c get_size_of_timing_pos().

    \warning This is currently not known yet for many non-TOF scanners. The function will 
    then throw an error.
  */
  float get_coincidence_window_width_in_ps() const;
  //! Get the full width of the coincidence window in millimeter
  /*! Calls get_coincidence_window_width_in_ps() */
  float get_coincidence_window_width_in_mm() const;

  //! \name number of "fake" crystals per block, inserted by the scanner
  /*! Some scanners (including many Siemens scanners) insert virtual crystals in the sinogram data.
    The other members of the class return the size of the "virtual" block. With these
    functions you can find its true size (or set it).

    You have to call set_up() after using the \c set_* functions.
  */
  //@{! 
  int get_num_virtual_axial_crystals_per_block() const;
  int get_num_virtual_transaxial_crystals_per_block() const;
  void set_num_virtual_axial_crystals_per_block(int);
  void set_num_virtual_transaxial_crystals_per_block(int);
  //@}

  //! \name functions to get block geometry info
  //@{
  //! get scanner geometry
  /*! \see set_scanner_geometry */
  inline std::string get_scanner_geometry() const;
  //! get crystal spacing in axial direction
  inline float get_axial_crystal_spacing() const;
  //! get crystal spacing in transaxial direction
  inline float get_transaxial_crystal_spacing() const;
  //! get block spacing in axial direction
  inline float get_axial_block_spacing() const;
  //! get block spacing in transaxial direction
  inline float get_transaxial_block_spacing() const;
  //@} (end of get block geometry info)
  
  //! \name functions to get generic geometry info
  //! get crystal map file name
  inline std::string get_crystal_map_file_name() const;
  
  //@} (end of block/bucket info)

  //@} (end of get geometrical info)

   //! \name Functions to get detector response info
  //@{

  //! get the energy resolution as a fraction at the reference energy
  /*! Values for PET scanners are around 0.1 at 511 keV, depending on the scanner of course.
    
    If less than or equal to 0, it is assumed to be unknown.
  */
  inline float get_energy_resolution() const;
  //! get the reference energy in keV of the energy resolution
  /*! For PET, normally set to 511 */
  inline float get_reference_energy() const;
  //! \c true if energy_resolution and reference_energy are set
  inline bool has_energy_information() const;

  //@} (end of get detector response info)

  //! \name Functions setting info
  /*! Be careful to keep consistency by setting all relevant parameters.

    You have to call set_up() after using any of these.
  */
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
  //! set inner ring radius
  inline void set_inner_ring_radius(const float & new_radius);
  //! set average depth of interaction
  inline void set_average_depth_of_interaction(const float& new_depth_of_interaction);
  //! set ring spacing 
  inline void set_ring_spacing(const float & new_spacing);
  //! set default arc-corrected bin size
  inline void set_default_bin_size(const float &new_size);
  //! in degrees
  inline void set_intrinsic_azimuthal_tilt(const float new_tilt);
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
  inline void set_num_detector_layers(const int& new_num);      
  //! set number of axial crystals per singles unit
  inline void set_num_axial_crystals_per_singles_unit(const int & new_num);     
  //! set number of transaxial crystals per singles unit
  inline void set_num_transaxial_crystals_per_singles_unit(const int & new_num);
  // TODO accomodate more complex geometries of singles units.
  //@{
  //! name functions to set block geometry info
  //! set scanner geometry
  /*! 
   \param new_scanner_geometry: "Cylindrical", "BlocksOnCylindrical" or "Generic"
    
    Will also read the detector map from file if the geometry is \c "Generic".
    \warning you need to call set_up() after calling this function.

   \see set_scanner_geometry
  */
  void set_scanner_geometry(const std::string& new_scanner_geometry);
  //! set crystal spacing in axial direction
  inline void set_axial_crystal_spacing(const float & new_spacing);
  //! set crystal spacing in transaxial direction
  inline void set_transaxial_crystal_spacing(const float & new_spacing);
  //! set block spacing in axial direction
  inline void set_axial_block_spacing(const float & new_spacing);
  //! set block spacing in transaxial direction
  inline void set_transaxial_block_spacing(const float & new_spacing);
  //! set crystal map file name for the generic geometry
  /*! \warning, data is not read yet. use set_scanner_geometry() after calling this function */
  inline void set_crystal_map_file_name(const std::string& new_crystal_map_file_name);
  //@} (end of block geometry info)

  //@} (end of block/bucket info)
  //! set the energy resolution of the system
  /*! \sa get_energy_resolution() */
  inline void set_energy_resolution(const float new_num);
  //! set the reference energy (in keV) of the energy resolution
  /*! \sa get_reference_energy() */
  inline void set_reference_energy(const float new_num);
  //! Set the maximum number of TOF bins.
  inline void set_max_num_timing_poss(int new_num);
  //! Set the delta t which correspnds to the max number of TOF bins.
  inline void set_size_of_timing_poss(float new_num);
  //! Set timing resolution
  inline void set_timing_resolution(float new_num_in_ps);
  //@} (end of set info)

  //@} (end of set info)
  
  //! Calculate a singles bin index from axial and transaxial singles bin coordinates.
  inline int get_singles_bin_index(int axial_index, int transaxial_index) const;

  //! Method used to calculate a singles bin index from
  //! a detection position.
  inline int get_singles_bin_index(const DetectionPosition<>& det_pos) const; 
 

  //! Get the axial singles bin coordinate from a singles bin.
  inline int get_axial_singles_unit(int singles_bin_index) const;

  //! Get the transaxial singles bin coordinate from a singles bin.
  inline int get_transaxial_singles_unit(int singles_bin_index) const;

  //! True if it is TOF compatible.
  inline bool is_tof_ready() const;
  
  //! Get the STIR detection position (det#, ring#, layer#) given the detection position id in the input crystal map
  // used in CListRecordSAFIR.inl for accessing the coordinates
  inline stir::DetectionPosition<> get_det_pos_for_index(const stir::DetectionPosition<> & det_pos) const;
  //! Get the Cartesian coordinates (x,y,z) given the STIR detection position (det#, ring#, layer#)
  // used in ProjInfoDataGenericNoArcCorr.cxx for accessing the coordinates
  inline stir::CartesianCoordinate3D<float> get_coordinate_for_det_pos(const stir::DetectionPosition<>& det_pos) const;
  //! Get the Cartesian coordinates (x,y,z) given the detection position id in the input crystal map
  inline stir::CartesianCoordinate3D<float> get_coordinate_for_index(const stir::DetectionPosition<>& det_pos) const;
  //! Find detection position at a coordinate
  // used  in ProjInfoDataGenericNoArcCorr.cxx for accessing the get_bin
  inline Succeeded
     find_detection_position_given_cartesian_coordinate(DetectionPosition<>& det_pos,
                                                        const CartesianCoordinate3D<float>& cart_coord) const;

  shared_ptr<const DetectorCoordinateMap> get_detector_map_sptr() const
  { return detector_map_sptr; }

private:
  bool _already_setup;
  Type type;
  std::list<std::string> list_of_names;
  int num_rings;                /* number of direct planes */
  int max_num_non_arccorrected_bins; 
  int default_num_arccorrected_bins; /* default number of bins */
  int num_detectors_per_ring;   

  float inner_ring_radius;      /*! detector inner radius in mm*/
  float average_depth_of_interaction; /*! Average interaction depth in detector crystal */
  float max_FOV_radius;       /*! detector maximum radius in mm - for cylindrical scanner identical to inner radius */
  float ring_spacing;   /*! ring separation in mm*/
  float bin_size;               /*! arc-corrected bin size in mm (spacing of transaxial elements) */
  float intrinsic_tilt;         /*! intrinsic tilt in radians*/

  int num_transaxial_blocks_per_bucket; /* transaxial blocks per bucket */
  int num_axial_blocks_per_bucket;      /* axial blocks per bucket */
  int num_axial_crystals_per_block;     /* number of crystals in the axial direction */
  int num_transaxial_crystals_per_block;/* number of transaxial crystals */
  int num_detector_layers;

  int num_axial_crystals_per_singles_unit;
  int num_transaxial_crystals_per_singles_unit;

   //!
  //! \brief energy resolution (FWHM as a fraction of the reference_energy)
  //! \details This is the energy resolution of the system.
  //! A negative value indicates, unknown.
  //! This value is dominated by the material of the scintilation crystal
  float energy_resolution;
  //! In PET application this should always be 511 keV.
  //! A negative value indicates, unknown.
  float reference_energy;
  //! The timing resolution of the scanner, in psec.
  float timing_resolution;
  //! The number of TOF bins. Without any mash factors
  int max_num_of_timing_poss;
  //! This number corresponds the the least significant clock digit.
  float size_timing_pos;

  //!
  //! \brief scanner info needed for block geometry
  //! \author Parisa Khateri
  //! A negative value indicates unknown.
  std::string scanner_geometry;          /*! scanner geometry */
  float axial_crystal_spacing;           /*! crystal pitch in axial direction in mm*/
  float transaxial_crystal_spacing;      /*! crystal pitch in transaxial direction in mm*/
  float axial_block_spacing;             /*! block pitch in axial direction in mm*/
  float transaxial_block_spacing;        /*! block pitch in transaxial direction in mm*/
  
  std::string crystal_map_file_name;
  shared_ptr<DetectorCoordinateMap> detector_map_sptr;  /*! effective detection positions including average DOI */

  void set_detector_map( const DetectorCoordinateMap::det_pos_to_coord_type& coord_map );
  void initialise_max_FOV_radius();

  // function to create the maps
  void read_detectormap_from_file( const std::string& filename );

  // ! set all parameters
  void set_params(Type type_v, const std::list<std::string>& list_of_names_v,
                  int num_rings_v,
                  int max_num_non_arccorrected_bins_v,
                  int default_num_arccorrected_bins_v,
                  int num_detectors_per_ring_v,
                  float inner_ring_radius_v,
                  float average_depth_of_interaction_v,
                  float ring_spacing_v,
                  float bin_size_v, float intrinsic_tilt_v,
                  int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
                  int num_axial_crystals_per_block_v, int num_transaxial_crystals_per_block_v,
                  int num_axial_crystals_per_singles_unit_v,
                  int num_transaxial_crystals_per_singles_unit_v,
                  int num_detector_layers_v,
                  float energy_resolution_v = -1.0f,
                  float reference_energy = -1.0f,
                  short int max_num_of_timing_poss_v = -1.0f,
                  float size_timing_pos_v = -1.0f,
                  float timing_resolution_v = -1.0f,
                  const std::string& scanner_geometry_v = "",
                  float axial_crystal_spacing_v = -1.0f,
                  float transaxial_crystal_spacing_v = -1.0f,
                  float axial_block_spacing_v = -1.0f,
                  float transaxial_block_spacing_v = -1.0f,
                  const std::string& crystal_map_file_name = "");

};

END_NAMESPACE_STIR

#include "stir/Scanner.inl"

#endif
 
