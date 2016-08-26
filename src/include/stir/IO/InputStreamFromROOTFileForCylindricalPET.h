#ifndef __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__
#define __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__


#include "stir/IO/InputStreamFromROOTFile.h"

class InputStreamFromROOTFileForCylindricalPET : public InputStreamFromROOTFile
{

public:

    virtual ~InputStreamFromROOTFileForCylindricalPET() {}

    virtual
    Succeeded get_next_record(CListRecordROOT& record);

private:
    Int_t submoduleID1, submoduleID2;
    Int_t moduleID1, moduleID2;

    int submodule_repeater_x;
    int submodule_repeater_y;
    int submodule_repeater_z;
    int module_repeater_x;
    int module_repeater_y;
    int module_repeater_z;

    //! In GATE, inside a block, the indeces start from the lower
    //! unit counting upwards. Therefore in order to align the
    //! crystals, between STIR and GATE we have to move half block more.
    int half_block;
};

Succeeded
InputStreamFromROOTFileForCylindricalPET::
get_next_record(CListRecordROOT& record)
{

    while(true)
    {
        if (current_position == nentries)
            return Succeeded::no;


        if (stream_ptr->GetEntry(current_position) == 0 )
            return Succeeded::no;

        current_position ++ ;

        if ( (comptonphantom1 > 0 && comptonphantom2>0) && exclude_scattered )
            continue;
        else if ( (eventID1 != eventID2) && exclude_randoms )
            continue;
        else if (energy1 < low_energy_window ||
                 energy1 > up_energy_window ||
                 energy2 < low_energy_window ||
                 energy2 > up_energy_window)
            continue;
        else
            break;
    }

    int ring1 = static_cast<int>(crystalID1/crystal_repeater_y)
            + static_cast<int>(submoduleID1/submodule_repeater_y)*crystal_repeater_z
            + static_cast<int>(moduleID1/module_repeater_y)*submodule_repeater_z*crystal_repeater_z;

    int ring2 = static_cast<int>(crystalID2/crystal_repeater_y)
            + static_cast<int>(submoduleID2/submodule_repeater_y)*crystal_repeater_z
            + static_cast<int>(moduleID2/module_repeater_y)*submodule_repeater_z*crystal_repeater_z;

    int crystal1 = rsectorID1  * module_repeater_y * submodule_repeater_y * crystal_repeater_y
            + (moduleID1%module_repeater_y) * submodule_repeater_y * crystal_repeater_y
            + (submoduleID1%submodule_repeater_y) * crystal_repeater_y
            + (crystalID1%crystal_repeater_y);

    int crystal2 = rsectorID2 * module_repeater_y * submodule_repeater_y * crystal_repeater_y
            + (moduleID2%module_repeater_y) * submodule_repeater_y * crystal_repeater_y
            + (submoduleID2% submodule_repeater_y) * crystal_repeater_y
            + (crystalID2%crystal_repeater_y);

    // GATE counts crystal ID =0 the most negative. Therefore
    // ID = 0 should be negative, in Rsector 0 and the mid crystal ID be 0 .
    crystal1 -= half_block;
    crystal2 -= half_block;

    // Add offset
    crystal1 += offset_dets;
    crystal2 += offset_dets;

    return
            record.init_from_data(ring1, ring2,
                                  crystal1, crystal2,
                                  time1, time2,
                                  eventID1, eventID2);
}

#endif
