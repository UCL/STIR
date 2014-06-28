//
//
/*!
  \file
  \ingroup DataProcessor
  \brief Inline implementations for class stir::DataProcessor

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

START_NAMESPACE_STIR

 
template <typename DataT>
DataProcessor<DataT>::
DataProcessor()
: is_set_up_already(false)
{}
   
template <typename DataT>
Succeeded 
DataProcessor<DataT>::
set_up(const DataT& image)
{
  start_timers();
  Succeeded result = virtual_set_up(image);
  is_set_up_already = (result == Succeeded::yes);
  stop_timers();
  return result;
}

template <typename DataT>
void
DataProcessor<DataT>::
reset()
{
  this->is_set_up_already = false;
}

template <typename DataT>
bool
DataProcessor<DataT>::
parse(std::istream& f)
{
  this->reset();
  return ParsingObject::parse(f);
}

template <typename DataT>
bool
DataProcessor<DataT>::
parse(const char * const filename)
{
  this->reset();
  return ParsingObject::parse(filename);
}



template <typename DataT>
Succeeded
DataProcessor<DataT>::
apply(DataT& data)
{
  start_timers();
  //assert(consistency_check(data) == Succeeded::yes);
  if (!is_set_up_already )
    if (set_up(data) == Succeeded::no)
      {
	warning("DataProcessor::apply: Building was unsuccesfull. No processing done.\n");
	return Succeeded::no;
      }
  virtual_apply(data);
  stop_timers();
  return Succeeded::yes;
}


template <typename DataT>
Succeeded
DataProcessor<DataT>::
apply(DataT& data,
      const DataT& in_data)
{
  start_timers();
  //assert(consistency_check(in_data) == Succeeded::yes);
  if (!is_set_up_already )
    if (set_up(in_data) == Succeeded::no)
      {
	warning("DataProcessor::apply: Building was unsuccesfull. No processing done.\n");
	return Succeeded::no;
      }
  virtual_apply(data, in_data);
  stop_timers();
  return Succeeded::yes;
}

#if 0
template <typename DataT>
Succeeded 
DataProcessor<DataT>::
consistency_check( const DataT& image) const
{
  return Succeeded::yes;
}
#endif

END_NAMESPACE_STIR
