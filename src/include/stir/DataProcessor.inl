//
// $Id$
//
/*!
  \file
  \ingroup DataProcessor
  \brief Inline implementations for class DataProcessor

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet
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
