# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:18:13 2021

@author: lgauthier

A time object handling the base 60 nature of minutes and seconds and the
base 24 nature of hours and able to represent time as a pure integer like
in many traffic count files (ie: 700 stands for 7h00 or 615 for 6h15).

To calculate the absolute difference between two arrays of Ctime object, use
the following strategy:

    TF = array1 >= arrays2
    Delta = arrays1 - arrays2

    anw = TF * Delta + (1 - TF) * (240000 - Delta)
"""
######################
# Top level imports
######################
import sys, math, operator
import datetime
import numpy as np
import pandas
import numbers

######################
# Utils
######################
def multireplace(text, replacedict, ignore_case=False):
    for key,value in replacedict.items():
        proceed=False
        if ignore_case:
            if key.lower() in text.lower():
                text = text.replace(key,value)
        elif key in text:
            text = text.replace(key,value)
    return text

def round_on(x, base=5):
    '''version of math.ceil that ceils to the nearest base multiple'''
    if isinstance(base, (int, np.integer)):
        return int(base * round((float(x)/base)))
    if isinstance(base, (float, np.inexact)):
        return base * round((float(x)/base))

def floor_on(x, base=5):
    '''version of math.ceil that ceils to the nearest base multiple'''
    if isinstance(base, (int, np.integer)):
        return int(base * math.floor((float(x)/base)))
    if isinstance(base, (float, np.inexact)):
        return base * math.floor((float(x)/base))

def ceil_on(x, base=5):
    '''version of math.ceil that ceils to the nearest base multiple'''
    if isinstance(base, (int, np.integer)):
        return int(base * math.ceil((float(x)/base)))
    if isinstance(base, (float, np.inexact)):
        return base * math.ceil((float(x)/base))

######################
# CustomTime
######################
class FormatError(Exception):
    def __init__(self, message, content=None):
        self.message = message
        self.content = content

        super().__init__(message)

class CTimeNull(object):
    """Null object -- most interactions returns Null
    """
    _instance = None

    def _nullEq(self, *args, **kwargs):
        if args[0] is None:
            return True
        elif isinstance(args[0], CTimeNull):
            return True
        return False

    def _null(self, *args, **kwargs):
        return None

    __eq__ = __ne__ = __ge__ = __gt__ = __le__ = __lt__ = _nullEq
    __add__ = __iadd__ = __radd__ = _null
    __sub__ = __isub__ = __rsub__ = _null
    __mul__ = __imul__ = __rmul__ = _null
    __div__ = __idiv__ = __rdiv__ = _null
    __mod__ = __imod__ = __rmod__ = _null
    __pow__ = __ipow__ = __rpow__ = _null
    __and__ = __iand__ = __rand__ = _null
    __xor__ = __ixor__ = __rxor__ = _null
    __or__ = __ior__ = __ror__ = _null
    __truediv__ = __itruediv__ = __rtruediv__ = _null
    __floordiv__ = __ifloordiv__ = __rfloordiv__ = _null
    __lshift__ = __ilshift__ = __rlshift__ = _null
    __rshift__ = __irshift__ = __rrshift__ = _null
    __neg__ = __pos__ = __abs__ = __invert__ = _null
    __call__ = __getattr__ = _null
    __getitem__ = _null

    def __divmod__(self, other):
        return self, self
    __rdivmod__ = __divmod__

    __hash__ = None

    def __new__(cls, *args, **kwargs):
        if CTimeNull._instance is None:
            CTimeNull._instance = object.__new__(cls, *args, **kwargs)
        return CTimeNull._instance

    def __bool__(self):
        return False

    def __repr__(self):
        return '<CTimeNull>'

    def __setattr__(self, name, value):
        return None

    def __setitem___(self, index, value):
        return None

    def __str__(self):
        return ''

class Ctime(object):
    """
    A time object handling the base 60 nature of minutes and seconds and the
    base 24 nature of hours and able to represent time as a pure integer like
    in many traffic count files (ie: 700 stands for 7h00 or 615 for 6h15).

    Note: there is a built-in wrap in behavior at the day, ie: at 23h59:59,
    adding a second returns the time back to 0h00:00 without returning that a
    day threshold was passed
    """

    def __init__(self, time=None, time_is_HHMM=False):
        """time as int with digits meaning: HHMMSS"""
        if time is None:
            #NB: allowing time=None makes the class able to be used with np.vectorize
            self._time = CTimeNull()
        else:
            if not isinstance(time, int):
                try:
                    time = int(time)
                except:
                    raise TypeError('expected int, received '+str(time.__class__.__name__))

            if time_is_HHMM:
                time = time * 100

            if time > 235959:
                time = time % 240000

            self._time = time

    @property
    def hhmm(self):
        return self._time // 100

    def __repr__(self):
        if self._time is CTimeNull(): return self._time.__repr__()
        return f"<Ctime: {self.hour:02}hrs:{self.minute:02}min:{self.second:02}sec>"

    def __eq__(self, other):
        if self._time is CTimeNull(): return self._time.__eq__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time == other._time:
                return True
            return False
        except:
            return False

    def __ne__(self, other):
        if self._time is CTimeNull(): return self._time.__ne__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        if self.__eq__(other):
            return False
        return True

    def __gt__(self, other):
        if self._time is CTimeNull(): return self._time.__gt__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time > other._time:
                return True
            return False
        except:
            return False

    def __ge__(self, other):
        if self._time is CTimeNull(): return self._time.__ge__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time >= other._time:
                return True
            return False
        except:
            return False

    def __lt__(self, other):
        if self._time is CTimeNull(): return self._time.__lt__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time < other._time:
                return True
            return False
        except:
            return False

    def __le__(self, other):
        if self._time is CTimeNull(): return self._time.__le__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time <= other._time:
                return True
            return False
        except:
            return False

    def __str__(self):
        if self._time is CTimeNull(): return self._time.__str__(other)
        return self.as_string()

    def __add__(self, other):
        if self._time is CTimeNull(): return self._time.__add__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        elif isinstance(other, str):
            pass

        try:
            if other._time < 0:
                return self.__sub__(Ctime(-1*other._time))

            hour1 = self.hour;    hour2 = other.hour
            min1  = self.minute;  min2  = other.minute
            sec1  = self.second;  sec2  = other.second

            add_hour = (hour1 + hour2 + (min1 + min2 + (sec1 + sec2) // 60) // 60) % 24 *10000
            add_min = (min1 + min2 + (sec1 + sec2) // 60) % 60 * 100
            add_sec = (sec1 + sec2) % 60

            return self.__class__(add_hour + add_min + add_sec)
        except:
            return NotImplemented

    def __iadd__(self, other):
        if self._time is CTimeNull(): return self._time.__iadd__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)

        try:
            if other._time < 0:
                return self.__sub__(Ctime(-1*other._time))

            hour1 = self.hour; hour2 = other.hour
            min1  = self.minute;  min2  = other.minute
            sec1  = self.second;  sec2  = other.second

            add_hour = (hour1 + hour2 + (min1 + min2 + (sec1 + sec2) // 60) // 60) % 24 *10000
            add_min = (min1 + min2 + (sec1 + sec2) // 60) % 60 * 100
            add_sec = (sec1 + sec2) % 60

            self._time = add_hour + add_min + add_sec

            return self
        except:
            return NotImplemented

    def __radd__(self, other):
        if self._time is CTimeNull(): return self._time.__radd__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            return self.__add__(other)
        except:
            return NotImplemented

    def __sub__(self, other):
        if self._time is CTimeNull(): return self._time.__sub__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)

        try:
            if other._time < 0:
                return self.__add__(Ctime(-1*other._time))

            hour1 = self.hour; hour2 = other.hour
            min1  = self.minute;  min2  = other.minute
            sec1  = self.second;  sec2  = other.second

            sub_hour = (hour1 - hour2 + (min1 - min2 + (sec1 - sec2) // 60) // 60) % 24 *10000
            sub_min = (min1 - min2 + (sec1 - sec2) // 60) % 60 * 100
            sub_sec = (sec1 - sec2) % 60

            return self.__class__(sub_hour + sub_min + sub_sec)
        except:
            return NotImplemented

    def __rsub__(self, other):
        if self._time is CTimeNull(): return self._time.__rsub__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            return other.__sub__(self)
        except:
            return NotImplemented

    def __mul__(self, other):
        if self._time is CTimeNull(): return self._time.__mul__(other)
        _self_time =  self.hour  * 3600 + self.minute  * 60 + self.second

        if _self_time == 0:  #log 0 will fail
            return self.__class__(0)

        if isinstance(other, numbers.Number) and not isinstance(other, bool):

            if other == 0:   #log 0 will fail
                return self.__class__(0)

            final_seconds = int(round(math.exp( math.log(_self_time) + math.log(other) ),0))
            secs = final_seconds % 60
            mins = (final_seconds // 60) % 60  * 100
            hour = (final_seconds // 60) // 60 * 10000

            return self.__class__(hour+mins+secs)

        try:
            if other._time == 0:   #log 0 will fail
                return self.__class__(0)

            _other_time = other.hour * 3600 + other.minute * 60 + other.second
            return Ctime(int(round(math.exp( math.log(_self_time) - math.log(_other_time) ),0)))
        except:
            return NotImplemented

    def __rmul__(self, other):
        if self._time is CTimeNull(): return self._time.__rmul__(other)
        return self.__mul__(other)

    def __truediv__(self, other):
        if self._time is CTimeNull(): return self._time.__truediv__(other)
        _self_time =  self.hour  * 3600 + self.minute  * 60 + self.second

        if _self_time == 0: #log 0 will fail
            return self.__class__(0)

        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            #Ctime / number returns a Ctime
            if other == 0:
                raise ZeroDivisionError('{} division by zero using {}'.format(self.__class__, other.__class__))

            final_seconds = int(round(math.exp( math.log(_self_time) - math.log(other) ),0))
            secs = final_seconds % 60
            mins = (final_seconds // 60) % 60  * 100
            hour = (final_seconds // 60) // 60 * 10000
            return self.__class__(hour+mins+secs)

        try:
            #Ctime / Ctime returns a float
            if other._time == 0:
                raise ZeroDivisionError('{} division by zero using {}'.format(self.__class__, other.__class__))

            _self_time =  self.hour * 3600 + self.minute * 60 + self.second
            _other_time = other.hour * 3600 + other.minute * 60 + other.second
            return math.exp( math.log(_self_time) - math.log(_other_time) )
        except:
            return NotImplemented

    def __rtruediv__(self, other):
        if self._time is CTimeNull(): return self._time.__rtruediv__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            return Ctime(int(round(other / float(self._time),0)))
        else:
            return self.__truediv__(other)

    def addTime(self, time):
        """

        Parameters
        ----------
        time : int


        Returns
        -------
		Adds an amount of time to the Ctime._time attribute
        """
        self._time = (self + Ctime(time))._time

    def as_datetime(self, time_only=True, date='today'):
        return ctime_to_datetime(self, time_only=time_only, date=date)

    def as_string(self, stringFormat='hhmmss', separate_time_segments=True, min_sec_separator=':', hour_min_separator=':', **kwargs):
        """stringFormat : 'hhmmss', 'hhmm', 'hmm', 'mmss'
            Format to return the time string as

        Parameters
        ----------
        stringFormat :
             (Default value = 'hhmmss')
        separate_time_segments :
             (Default value = True)
        min_sec_separator :
             (Default value = ':')
        hour_min_separator :
             (Default value = ':')

        Returns
        -------

        """
        if stringFormat == 'gtfs':
            stringFormat = 'hhmmss'
            separate_time_segments=False
            min_sec_separator=':'
            hour_min_separator=':'

        if stringFormat == 'hhmmss':
            if separate_time_segments:
                return '{:02d}{}{:02d}{}{:02d}'.format(self.hour, hour_min_separator, self.minute, min_sec_separator, self.second)
            else:
                return '{:02d}{:02d}{:02d}'.format(self.hour, self.minute, self.second)

        if stringFormat == 'hhmm':
            if separate_time_segments:
                return '{:02d}{}{:02d}'.format(self.hour, hour_min_separator, self.minute)
            else:
                return '{:02d}{:02d}'.format(self.hour, self.minute)

        if stringFormat == 'hmm':
            if separate_time_segments:
                return '{:01d}{}{:02d}'.format(self.hour, hour_min_separator, self.minute)
            else:
                return '{:01d}{:02d}'.format(self.hour, self.minute)

        if stringFormat == 'mmss':
            if separate_time_segments:
                return '{:02d}{}{:02d}'.format(self.minute, min_sec_separator, self.second)
            else:
                return '{:02d}{:02d}'.format(self.minute, self.second)

        else:
            raise FormatError('Unrecognised format')

    @property
    def time(self):
        if self._time is CTimeNull(): return None
        return self._time

    @property
    def second(self):
        return self._get_sec()

    @property
    def minute(self):
        return self._get_min()

    @property
    def hour(self):
        return self._get_hour()

    @time.setter
    def time(self, value):
        self._time = Ctime(value)

    @second.setter
    def second(self, value):
        hhmm = (self._time // 100) * 100
        self._time = (Ctime(hhmm) + Ctime(value))._time

    @minute.setter
    def minute(self, value):
        hh = (self._time // 10000) * 10000
        ss = self._time % 100
        self._time = (Ctime(hh) + Ctime(value * 100) + Ctime(ss))._time

    @hour.setter
    def hour(self, value):
        mmss = self._time % 10000
        self._time = (Ctime(value * 10000) + Ctime(mmss))._time

    def _get_sec(self):
        """

        Returns
        -------
		Amount of seconds as int
        """

        return self._time % 100

    def get_total_sec(self):
        return self.hour * 3600 + self.minute * 60 + self.second

    def get_total_min(self, add_sec_as_float=False):
        if not add_sec_as_float:
            return self.hour * 60 + self.minute
        else:
            return self.hour * 60 + self.minute + self.second / 60.0

    def get_total_hour(self, add_min_sec_as_float=False):
        if not add_min_sec_as_float:
            return self.hour
        else:
            return self.hour + self.minute/60.0 + self.second / 3600.0

    def _get_min(self):
        """

        Returns
        -------
		Amount of minutes as int
        """
        return (self._time // 100) % 100

    def _get_hour(self):
        """

        Returns
        -------
		Amount of hours as int
        """
        return self._time // 10000

    def round_time(self, round_unit='min', how='round', round_min_base=None, as_string=False, stringFormat='hhmmss', strFormatKwargs={}):
        return round_ctime(self, round_unit=round_unit, how=how,
                           round_min_base=round_min_base, as_string=as_string,
                           stringFormat=stringFormat, strFormatKwargs=strFormatKwargs)

    @staticmethod
    def from_string(time, hour_format=None, separator=':', return_hour_format=False):
        """Builds a Ctime object from a str object formatted into standard time formats

        Parameters
        ----------
        time : str

        hour_format : str

			The str hour_format can be any of those types:
            - 24 hour formats:
                HH:MM:SS       example: 13:05:32)
                HH:MM          example: 13:05

            - 12 hour formats:
                12H HH:MM:SS   example: 01:05:32 PM
                12H HH:MM      example: 01:05 PM

            - None:
                will automatically try to detect the format, including the
                separator

            (Default value = None)

        separator : str
            The separator used in the string to delimit hours, minutes and
            seconds components.
            (Default value = ':')

        return_hour_format : Bool (optional)
            if True, the output will contain the format used

        Returns
        -------
		Ctime object (and format used if selected)
        """
        #TODO: add mm:ss format


        if hour_format is None:

            #first try to determine the separator
            possible_separators = [':', 'h', ' ']
            #the space is problematic is mixed with a space before the AM/PM tag
            time = multireplace(time, {item:':' for item in possible_separators if item != ' '}, ignore_case=True)
            #check to replace the space with a more rubst algorithm
            if 'AM' in time.upper():
                time = time.upper().replace('AM', '').strip().replace(' ',separator)+' AM'
            elif 'PM' in time.upper():
                time = time.upper().replace('PM', '').strip().replace(' ',separator)+' PM'
            else:
                pass

            if time.lower().count(separator) == 0:
                if 'AM' in time.upper() or 'PM' in time.upper():
                    hour_format = '12H HH'
                else:
                    hour_format = 'HH'

            if time.lower().count(separator) == 1:
                if 'AM' in time.upper() or 'PM' in time.upper():
                    hour_format = '12H HH:MM'
                else:
                    hour_format = 'HH:MM'

            if time.lower().count(separator) == 2:
                if 'AM' in time.upper() or 'PM' in time.upper():
                    hour_format = '12H HH:MM:SS'
                else:
                    hour_format = 'HH:MM:SS'


        time = time.replace(separator,':')

        if hour_format == 'HH:MM:SS':
            ctime = Ctime(int(time.replace(':','')))

        elif hour_format == 'HH:MM':
            ctime = Ctime(int(time.replace(':',''))*100)

        elif hour_format == 'HH':
            ctime = Ctime(int(time.replace(':',''))*10000)

        elif hour_format == '12H HH:MM:SS':
            if 'AM' in time:
                ctime = Ctime.from_string(time.replace('AM','').strip(), hour_format='HH:MM:SS')
            if 'PM' in time:
                ctime = Ctime.from_string(time.replace('PM','').strip(), hour_format='HH:MM:SS') + Ctime(120000)

        elif hour_format == '12H HH:MM':
            if 'AM' in time.upper():
                ctime = Ctime.from_string(time.replace('AM','').strip(), hour_format='HH:MM')
                if ctime._time == 120000:
                    ctime = Ctime(0)

            if 'PM' in time.upper():
                ctime = Ctime.from_string(time.replace('PM','').strip(), hour_format='HH:MM')
                if ctime._time < 120000:
                    ctime = ctime + 120000

        elif hour_format == '12H HH':
            if 'AM' in time.upper():
                ctime = Ctime.from_string(time.replace('AM','').strip(), hour_format='HH')
                if ctime._time == 120000:
                    ctime = Ctime(0)

            if 'PM' in time.upper():
                ctime = Ctime.from_string(time.replace('PM','').strip(), hour_format='HH')
                if ctime._time < 120000:
                    ctime = ctime + 120000

        if return_hour_format:
            return ctime, hour_format
        else:
            return ctime

    @staticmethod
    def from_declared_times(hours=0, minutes=0, seconds=0):
        """
		Builds a Ctime objects from a given amount of hours, minutes and seconds

        Parameters
        ----------
        hours : int or float
             If a float is passed, the floating part will be added to the
             minutes as a fraction of 60 minutes (ie: 5.80 hours = 5 hours and
             48 minutes). For the purpose of float checking, the `hours`
             parameter is processed before the `minutes` parameter.
             (Default value = 0)

        minutes : int or float
             If a float is passed, the floating part will be added to the
             seconds as a fraction of 60 seconds (ie: 5.80 minutes = 5 minutes
             and 48 seconds). For the purpose of float checking, the `minutes`
             parameter is processed before the `seconds` parameter.
             (Default value = 0)

        seconds : int or float
             If a float is passed, the amount is rounded to the nearest integer.
             (Default value = 0)

        Returns
        -------
		Ctime object
        """
        if hours % 1 > 0:
            minutes += hours % 1 * 60
            hours = hours // 1

        if minutes % 1 > 0:
            seconds += minutes % 1 * 60
            minutes = minutes // 1

        if seconds  % 1 > 0:
            seconds = round(seconds,0)

        _hour = (hours + (minutes + (seconds) // 60) // 60) % 24 *10000
        _min = (minutes + (seconds) // 60) % 60 * 100
        _sec = (seconds) % 60

        return Ctime(_hour + _min + _sec)

    @staticmethod
    def from_list(times, method='from_string', **kwargs):
        """
        This method accepts a list of inputs

        Parameters
        ----------
        times : list of string or dict objects (see description)

        method : str or array-like
             The method used to read the elements of times.
             Value is either `from_string` or `from_declared_times`
             (Default value = from_string)

             For the from_string method, times can contain strings, with the
             format specified or to be automatically detected by the method.
             If a format is specified, it will be assumed for evey item of
             times.

             For the from_declared_times, times must be dictionnaries
             containing at least one of the following keywords: `hours`,
             `minutes`, or `seconds` and their corresponding values.
             (ex: {hours=13, minutes=58}, {hours=13, minutes=58, seconds=25})

             If different methods must be used for the different elements,
             pass those methods in an array-like object of equal lenght to
             times.

        Returns
        -------
		list of Ctime objects
        """

        if isinstance(method, str):
            method = [method for i in range(len(times))]

        out=[]
        for time, meth in zip(times, method):
            if meth == 'from_string':
                if 'return_hour_format' in kwargs.keys():
                    kwargs['return_hour_format'] = False
                out.append(Ctime.from_string(time, **kwargs))

            if meth == 'from_declared_times':
                out.append(Ctime.from_declared_times(hours=time['hours'] if 'hours' in time.keys() else 0,
                                                     minutes=time['minutes'] if 'minutes' in time.keys() else 0,
                                                     seconds=time['seconds'] if 'seconds' in time.keys() else 0)
                           )

        return out

    @staticmethod
    def from_datetime(time):
        if isinstance(time, np.datetime64):
            #note: we treat it this way so we can vectorize it to call on a pandas df
            #in vector form, screw trying to maintain this shit myself
            time = pandas.Timestamp(time)
        return Ctime.from_declared_times(hours=time.hour, minutes=time.minute, seconds=time.second)

    def isbetween(self, min_ctime, max_ctime, include_bot=False, include_top=False):
        if include_bot:
            bot_eq = operator.ge
        else:
            bot_eq = operator.gt

        if include_top:
            top_eq = operator.le
        else:
            top_eq = operator.lt

        return bot_eq(self, min_ctime) and top_eq(self, max_ctime)

def int_to_timestring(time):
    """
    Converts an interger to a formatted string

    Parameters
    ----------
    time : int of format Ctime._time (HHMMSS)


    Returns
    -------
    Ctime object
    """
    return '{:d}h{:02d}:{:02d}'.format(time // 10000, (time // 100) % 100, time, time % 100)

def int_to_GTFS_timestring(time):
    """
    Converts an interger to a GTFS formatted string

    Parameters
    ----------
    time : int of format Ctime._time (HHMMSS)


    Returns
    -------
    str of GTFS format (HH:MM:SS)
    """
    return '{:02d}:{:02d}:{:02d}'.format(time // 10000, (time // 100) % 100, time, time % 100)

def mean_of_Ctimes(ctime_list):
    """
    Calculates the mean time of a list of ctime objects, circumventing the
    overflowing behavior (23h59:59 + 1 ==> 0h00:00) that make a normal sum
    and division combo fail when the sum of individual times make more than
    a full day

    Parameters
    ----------
    ctime_list : list of Ctime objects

    Returns
    -------
    Ctime object
    """
    total_seconds = sum([item.get_total_sec() for item in ctime_list if item._time is not CTimeNull() ])

    return Ctime.from_declared_times(seconds = float(total_seconds)/len(ctime_list))

def signed_ctime_sub(ctime1, ctime2, **kwargs):
    '''
    Since Ctime does not support negative times, this method can be used to
    get a string containing negative timestamps

    ex: In [1]: _24h = Ctime(240000)
        In [2]: _15min = Ctime(1500)

        In [3]: Ctime.ctime1_minus_ctime2_with_sign(_24h, _15min)
        Out[3]: '-23:45:00'

    Parameters
    ----------
    ctime1: a Ctime object

    ctime2: a Ctime object

    **kwargs: from as_string method arg list

    Returns
    -------
    string from as_string method
    '''
    if ctime1._time is CTimeNull() or ctime1._time is CTimeNull(): return ''
    if ctime1 <= ctime2:
        return (ctime1 - ctime2).as_string(**kwargs)
    else:
        return '-'+(ctime1 - ctime2).as_string(**kwargs)

def ctime_to_datetime(ctime, time_only=True, date='today'):
    if ctime._time is CTimeNull(): return None
    if time_only:
        return datetime.time(hour=ctime.hour, minute=ctime.minute, second=ctime.second)

    if date == 'today':
        td = datetime.datetime.today()
        return datetime.datetime(year=td.year, month=td.month, day=td.day,
                                 hour=ctime.hour, minute=ctime.minute, second=ctime.second)
    else:
        if not isinstance(date, dict):
            raise NotImplemented
        return datetime.datetime(year=date['year'], month=date['month'], day=date['day'],
                                 hour=ctime.hour, minute=ctime.minute, second=ctime.second)

def round_ctime(ctime, round_unit='min', how='round', round_min_base=None, as_string=False, stringFormat='hhmmss', strFormatKwargs={}):
    """

    Parameters
    ----------
    time: Ctime

    round_unit : the unit where the rounding takes place {'hour', 'min'}
         (Default value = 'min')
    round_to_15min : if True, returns time rounded on the closest 15 minutes interval, `ceil_unit` is ignored.
         (Default value = False)
    as_string : returns the result as a str using Ctime.as_string method
         (Default value = False)
    stringFormat : f as_string is True, the format used for the string
         (Default value = 'hhmmss')

    Returns
    -------
    Rounded time as int

    """

    if ctime._time is CTimeNull(): return None

    if how == 'round':
        func = round_on
    elif how == 'ceil':
        func = ceil_on
    elif how == 'floor':
        func = floor_on

    _hour = ctime.hour
    _mins = ctime.minute
    _secs = ctime.second

    if round_min_base is not None:
        ctime = Ctime(func(func(_secs, 60) // 60 + _mins, round_min_base) * 100) + Ctime(_hour * 10000)

    else:
        if round_unit == 'min':
            ctime = Ctime(func(_secs, 60)) + Ctime(_mins * 100) + Ctime(_hour * 10000)

        if round_unit == 'hrs':
            ctime = Ctime(func(func(_secs, 60) // 60 + _mins, 60) * 100) + Ctime(_hour * 10000)

    if as_string:
        return ctime.as_string(stringFormat=stringFormat, **strFormatKwargs)
    else:
        return ctime

if __name__ == "__main__":
    pass