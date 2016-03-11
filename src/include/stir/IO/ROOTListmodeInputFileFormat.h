#ifndef __stir_IO_ROOTListmodeInputFileFormat_h__
#define __stir_IO_ROOTListmodeInputFileFormat_h__


#include "stir/IO/InputFileFormat.h"
#include "stir/listmode/CListModeDataROOT.h"


#include "stir/utilities.h"
#include <string>

START_NAMESPACE_STIR

//!
//! \brief The ROOTListmodeInputFileFormat class
//! \details Class for being able to read list mode data from the ROOT via the listmode-data registry.
//! \author Nikos Efthimiou
//!
class ROOTListmodeInputFileFormat :
        public InputFileFormat<CListModeData >
{
public:
    virtual const std::string
    get_name() const
    {  return "ROOT"; }

protected:

    virtual
    bool
    actual_can_read(const FileSignature& signature,
                    std::istream& input) const
    {
        // TODO need to do check that it's a siemens list file etc
        // N.E. I keep to check if it is an interfile, but added the check
        // whether it is has a ROOT input.

        if (is_interfile_signature(signature.get_signature()))
            return ( has_root_input_file(signature.get_signature()));
        else
            return false;
    }

    //!
    //! \brief has_root_input_file
    //! \param signature
    //! \return
    //! \author Nikos Efthimiou
    //! \todo This could be written nicer
    //!
    bool
    has_root_input_file(const char* const signature) const
    {
        // checking for ".root" or ".ROOT"
        bool val = false;
        std::string str1 = (".root");
        std::string str2 = (".ROOT");

        std::string sig(signature);

        if ( sig.find(str1) == std::string::npos && sig.find(str2) == std::string::npos)
            val = false;
        else
            val = true;

        return val;
    }

public:
    virtual std::auto_ptr<data_type>
    read_from_file(std::istream& input) const
    {
        warning("read_from_file for ROOT listmode data with istream not implemented %s:%s. Sorry",
                __FILE__, __LINE__);
        return
                std::auto_ptr<data_type>
                (0);
    }

    virtual std::auto_ptr<data_type>
    read_from_file(const std::string& filename) const
    {
        return std::auto_ptr<data_type>(new CListModeDataROOT(filename));
    }



};

END_NAMESPACE_STIR

#endif
