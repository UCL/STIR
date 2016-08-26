#ifndef __stir_IO_InputStreamFromROOTFileForECATPET_H__
#define __stir_IO_InputStreamFromROOTFileForECATPET_H__

class InputStreamFromROOTFileForECATPET : public InputStreamFromROOTFile
{

public:


    virtual ~InputStreamFromROOTFileForECATPET() {}

    virtual
    Succeeded get_next_record(CListRecordROOT& record);

protected:


};

Succeeded
InputStreamFromROOTFileForECATPET::
get_next_record(CListRecordROOT& record)
{

}

#endif
