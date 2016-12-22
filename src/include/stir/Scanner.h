/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2010, Hammersmith Imanet Ltd
    Copyright (C) 2011-2013, King's College London
    Copyright (C) 2016, UCL
    Copyright (C) 2016, University of Hull
 
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
  \author PARAPET project
*/
#ifndef __stir_buildblock_SCANNER_H__
#define __stir_buildblock_SCANNER_H__

#include "stir/DetectionPosition.h"
#include <string>
#include <list>

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

      \warning This information is only sensible for discrete detector-based scanners.
      \warning Currently, in a TOF compatible scanner template, the last three types have to
                be explicitly defined to avoid ambiguity.
      \todo Some scanners do not have all info filled in at present. Values are then
      set to 0.

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
  static Scanner * get_scanner_from_name(const std::string& name);
  //! get the list of all names for the particular scanner
  static std::string list_all_names();

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
  enum Type {E931, E951, E953, E921, E925, E961, E962, E966, E1080, Siemens_mMR, RPT,HiDAC,
         /*Type_mCT, Type_mCT_TOF, Type_mCT_TOF_100, Type_mCT_TOF_200, Type_mCT_TOF_400,*/
	     Advance, DiscoveryLS, DiscoveryST, DiscoverySTE, DiscoveryRX, Discovery600,
	     HZLR, RATPET, PANDA, HYPERimage, nanoPET, HRRT, Allegro, GeminiTF, User_defined_scanner,
	     Unknown_scanner};
  
  //! constructor that takes scanner type as an input argument
  Scanner(Type scanner_type);


  //! constructor -(list of names)
  /*! size info is in mm
      \param intrinsic_tilt_v value in radians, \see get_default_intrinsic_tilt()
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
          float reference_energy_v = -1.0f);

  //! constructor ( a single name)
  /*! size info is in mm
      \param intrinsic_tilt value in radians, \see get_default_intrinsic_tilt()
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
          float reference_energy_v = -1.0f);


  //! TOF constructor -(list of names)
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
          int max_num_of_timing_bins,
          float size_timing_bin,
          float timing_resolution,
          float energy_resolution_v = -1.0f,
          float reference_energy_v = -1.0f);

  //! TOF constructor ( a single name)
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
          int max_num_of_timing_bins,
          float size_timing_bin,
          float timing_resolution,
          float energy_resolution_v = -1.0f,
          float reference_energy_v = -1.0f);



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

      \todo we still have to decide if ProjDataInfo::get_phi() will take 
      this tilt into account or not. At present, STIR ignores the intrinsic tilt.
  */
  inline float get_default_intrinsic_tilt() const;
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
  inline int get_num_max_of_timing_bins() const;
  //! Get the delta t which correspnds to the max number of TOF bins in picosecs.
  inline float get_size_of_timing_bin() const;
  //! Get the timing resolution of the scanner.
  inline float get_timing_resolution() const;

  //@} (end of block/bucket info)

  //@} (end of get geometrical info)

   //! \name Functions to get detector responce info
  //@{

  //! get the energy resolution of the system
  inline float get_energy_resolution() const;
  //! get the reference energy of the energy resolution
  inline float get_reference_energy() const;

  //@} (end of get detector responce info)

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
  //! set inner ring radius
  inline void set_inner_ring_radius(const float & new_radius);
  //! set average depth of interaction
  inline void set_average_depth_of_interaction(const float& new_depth_of_interaction);
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
  inline void set_num_detector_layers(const int& new_num);      
  //! set number of axial crystals per singles unit
  inline void set_num_axial_crystals_per_singles_unit(const int & new_num);     
  //! set number of transaxial crystals per singles unit
  inline void set_num_transaxial_crystals_per_singles_unit(const int & new_num);
  // TODO accomodate more complex geometries of singles units.

  //@} (end of block/bucket info)
  //! set the energy resolution of the system
  //! A negative value indicates, unknown || not set
  inline void set_energy_resolution(const float new_num);
  //! set the reference energy of the energy resolution
  //! A negative value indicates, unknown || not set
  inline void set_reference_energy(const float new_num);
  //! Set the maximum number of TOF bins.
  inline void set_num_max_of_timing_bins(int new_num);
  //! Set the delta t which correspnds to the max number of TOF bins.
  inline void set_size_of_timing_bin(float new_num);
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
  

private:
  Type type;
  std::list<std::string> list_of_names;
  int num_rings;                /* number of direct planes */
  int max_num_non_arccorrected_bins; 
  int default_num_arccorrected_bins; /* default number of bins */
  int num_detectors_per_ring;   

  float inner_ring_radius;      /*! detector inner radius in mm*/
  float average_depth_of_interaction; /*! Average interaction depth in detector crystal */
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
  //! \brief energy_resolution
  //! \author Nikos Efthimiou
  //! \details This is the energy resolution of the system.
  //! A negative value indicates, unknown.
  //! This value is dominated by the material of the scintilation crystal
  float energy_resolution;

  //!
  //! \brief reference_energy
  //! \author Nikos Efthimiou
  //! \details In PET application this should always be 511 keV.
  //! A negative value indicates, unknown.
  float reference_energy;

  //!
  //! \brief timing_resolution
  //! \author Nikos Efthimiou
  //! \details The timing resolution of the scanner, in psec.
  float timing_resolution;

  //!
  //! \brief num_tof_bins
  //! \author Nikos Efthimiou
  //! \details The number of TOF bins. Without any mash factors
  int max_num_of_timing_bins;

  //!
  //! \brief size_timing_bin
  //! \author Nikos Efthimiou
  //! \details This number corresponds the the least significant clock digit.
  float size_timing_bin;


  //! set all parameters, case where default_num_arccorrected_bins==max_num_non_arccorrected_bins
  void set_params(Type type_v, const std::list<std::string>& list_of_names_v,
                  int num_rings_v, 
                  int max_num_non_arccorrected_bins_v,
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
                  float reference_energy = -1.0f);

  //! set all parameters
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
                  float reference_energy = -1.0f);


};

END_NAMESPACE_STIR

#include "stir/Scanner.inl"

#endif
 
