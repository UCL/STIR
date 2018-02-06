#include "ExamInfo.h"

START_NAMESPACE_STIR

void
ExamInfo::set_low_energy_thres(float new_val)
{
    low_energy_thres = new_val;
}

void
ExamInfo::set_high_energy_thres(float new_val)
{
    up_energy_thres = new_val;
}

float
ExamInfo::get_low_energy_thres() const
{
    return low_energy_thres;
}

float
ExamInfo::get_high_energy_thres() const
{
    return up_energy_thres;
}

END_NAMESPACE_STIR
