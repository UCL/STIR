/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamWithRecordsFromUPENN

  \author Nikos Efthimiou

*/
/*
    Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_IO_InputStreamWithRecordsFromUPENN_H__
#define __stir_IO_InputStreamWithRecordsFromUPENN_H__

#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/RegisteredObject.h"
#include "stir/listmode/CListRecordPENN.h"

START_NAMESPACE_STIR

/*!
  \brief Base class for reading listmode files from the PENNPet Explorer scanner.
  \ingroup IO

  \par abrupt_stop is a lower level counter to make the listmode operations be more consistent on
  how they process the total number of events. For example, if a file has 1M events (prompts + delay)
  and the user wishes to process 50% of the file. If in the e.g. lm_to_projdata sets 500K then each type of
  event (either prompt or delay) will continue until 500K events have been processed. This breaks the randoms'
  fraction. Setting the abrupt_stop to 500K, the InputStream will stop on the 500K^{th} event of either type.

  \par Requirements
    \c This needs access to the IRX libraries.
  */
class InputStreamWithRecordsFromUPENN: public RegisteredObject< InputStreamWithRecordsFromUPENN >
{
public:
    typedef std::vector<long long int>::size_type SavedPosition;

    InputStreamWithRecordsFromUPENN();

    unsigned long int get_total_number_of_events(CListRecordPENN& record);

    virtual inline Succeeded
    create_output_file(const std::string ofilename) = 0;

    virtual ~InputStreamWithRecordsFromUPENN() {}
    //! Must be called before calling for the first event.
    virtual Succeeded set_up();

    virtual
    Succeeded get_next_record(CListRecordPENN& record) = 0;

    //! go back to starting position
    virtual Succeeded reset() = 0;

    virtual SavedPosition save_get_position() = 0;

    virtual
    Succeeded set_get_position(const SavedPosition&) = 0;

    inline
    std::vector<std::streampos> get_saved_get_positions() const;

    inline
    void set_saved_get_positions(const std::vector<std::streampos>& );

    inline
    std::streambuf& get_stream(){/*return &this->inputList;*/}

//    virtual const PET::ListFileHeader* get_file_header() const = 0;

    virtual void set_current_record() = 0;


    virtual void set_new_record(const bool& d,
                               const short int& _dt,
                               const unsigned short int& _xa, const unsigned short int& _xb,
                               const unsigned short int& _za, const unsigned short int& _zb,
                               const unsigned short int& _ea, const unsigned short int& _eb) = 0;

protected:
    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    std::string filename;
    std::streampos starting_stream_position;
    std::vector<std::streampos> saved_get_positions;

    //    uint8_t current_record[8];
    const uint8_t* current_record;

    int eventSize = 0;

    int low_energy_window = 0;
    int up_energy_window = 1000;

    int timeout = 0;
    //! Total number of events
    long unsigned int N;
    //! Stop after a predefined number of records, regardless of their type.
    long unsigned int abrupt_counter;
    //! This is a flag about a low lever function that replicates a listmode file preserving
    //! control records that are skipped in normal operations.
    bool has_output = false;
    //! This is a lower counter to abruptly stop the listmode file.
    bool abrupt_stop;
    //! Minimum energy channel
    int minE_chan;
    //! Maximum energy channel
    int maxE_chan;
    //! In some processes it might be better to filter prompts and delayeds
    //! early.
    bool keep_prompt;
    //! In some processes it might be better to filter prompts and delayeds
    //! early.
    bool keep_delayed;
};

END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecordsFromUPENN.inl"

#endif
