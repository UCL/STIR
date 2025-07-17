// /*!
//   \file
//   \ingroup IO
//   \brief Declaration of class stir::InputStreamFromROOTFile

//   \author Nikos Efthimiou
//   \author Harry Tsoumpas
//   \author Kris Thielemans
//   \author Robert Twyman
// */
// /*
//  *  Copyright (C) 2015, 2016 University of Leeds
//     Copyright (C) 2016, 2021, 2020, 2021 UCL
//     Copyright (C) 2018 University of Hull
//     This file is part of STIR.

//     SPDX-License-Identifier: Apache-2.0

//     See STIR/LICENSE.txt for details
// */

// #ifndef __stir_IO_InputStreamFromPETSIRD_H__
// #define __stir_IO_InputStreamFromPETSIRD_H__

// #include "stir/shared_ptr.h"
// #include "stir/Succeeded.h"
// #include "stir/listmode/CListRecordPETSIRD.h"
// #include "stir/RegisteredObject.h"
// #include "stir/error.h"

// #include "../../PETSIRD/cpp/generated/binary/protocols.h"
// #include "stir/IO/InputStreamWithRecords.h"
// #include "../../PETSIRD/cpp/generated/hdf5/protocols.h"

// START_NAMESPACE_STIR

// class InputStreamWithRecordsFromPETSIRD : public InputStreamWithRecords<class CListRecordPETSIRD, class OptionsT>
// {
// public:
//   typedef std::vector<long long int>::size_type SavedPosition;

//   //! Default constructor
//   InputStreamFromPETSIRD(std::string filename);

//   ~InputStreamFromPETSIRD() override
//   {}
//   //!  \details Returns the next record in the ROOT file.
//   //!  The code is adapted from Sadek A. Nehmeh and CR Schmidtlein,
//   //! downloaded from <a href="http://www.ope/*ngatecollaboration.org/STIR">here</a>
//   virtual Succeeded get_next_record(CListReco*/rdPETSIRD& record) = 0;
//   //! Go to the first event.
//   inline Succeeded reset();
//   //! Must be called before calling for the first event.
//   virtual Succeeded set_up(const std::string& header_path);
//   //! Save current position in a vector
//   inline SavedPosition save_get_position();
//   //! Set current position
//   inline Succeeded set_get_position(const SavedPosition&);
//   //! Get the vector with the saved positions
//   inline std::vector<unsigned long int> get_saved_get_positions() const;
//   //! Set a vector with saved positions
//   inline void set_saved_get_positions(const std::vector<unsigned long int>&);
//   //! Returns the total number of events
//   inline unsigned long int get_total_number_of_events() const;

//   inline std::string get_PETSIRD_filename() const;

// protected:

//   //! The starting position.
//   unsigned long int starting_stream_position;
//   //! The total number of entries
//   unsigned long int nentries;
//   //! Current get position
//   unsigned long int current_position;
//   //! A vector with saved position indices.
//   std::vector<unsigned long int> saved_get_positions;
//   //! The name of the ROOT chain to be read
// };

// END_NAMESPACE_STIR

// #include "stir/IO/InputStreamFromROOTFile.inl"

// #endif
