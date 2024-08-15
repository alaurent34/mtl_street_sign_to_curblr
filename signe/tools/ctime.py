# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:18:13 2021
Modified on Mon Jun 12 10:15:00 2024

@author: lgauthier
@author: alaurent

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
import math
import operator
import datetime
import numbers
import re
import numpy as np
import pandas


######################
# Utils
######################


def multireplace(
    text: str,
    replacedict: dict[str, str],
    ignore_case: bool = False
) -> str:
    """
    A function chaining the text replacements to avoid multiple calls. Pay
    attention to the order of replacements since they are done in order of key
    calling: previous replacements could invalide a later one if the pattern
    is changed.

    Parameters
    ----------
    text : string
        The text in which to perform replacements.
    replacedict : dict[str, str]
        DESCRIPTION.
    ignore_case : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    text : string
        DESCRIPTION.

    """
    for key, value in replacedict.items():
        if ignore_case:
            flag = re.IGNORECASE
        else:
            flag = None
        text = re.sub(key, value, text, flags=flag)
    return text


def round_on(x: int | float, base: int | float = 5) -> int | float:
    """
    A version of round that rounds to the nearest base multiple

    Parameters
    ----------
    x : int | float
        The number to round.
    base : int | float, optional
        The base on which to round. The default is 5.

    Returns
    -------
    int | float
        The rounded number.

    Raises
    ------
    ValueError
    """
    if (
        not isinstance(base, (int, np.integer)) or
        not isinstance(base, (float, np.inexact))
    ):
        raise ValueError('base must be int or float.')

    ans = base * round((float(x)/base))
    if isinstance(base, (int, np.integer)):
        return int(ans)
    return ans


def floor_on(x: int | float, base: int | float = 5) -> int | float:
    """
    A version of math.floor that floors to the nearest base multiple

    Parameters
    ----------
    x : int | float
        The number to floor.
    base : int | float, optional
        The base on which to floor. The default is 5.

    Returns
    -------
    int | float
        The floored number.

    Raises
    ------
    ValueError
    """
    if (
        not isinstance(base, (int, np.integer)) or
        not isinstance(base, (float, np.inexact))
    ):
        raise ValueError('base must be int or float.')

    ans = base * math.floor((float(x)/base))
    if isinstance(base, (int, np.integer)):
        return int(ans)
    return ans


def ceil_on(x: int | float, base: int | float = 5) -> int | float:
    """
    A version of math.ceil that ceils to the nearest base multiple

    Parameters
    ----------
    x : int | float
        The number to ceil.
    base : int | float, optional
        The base on which to ceil. The default is 5.

    Returns
    -------
    int | float
        The ceiled number.

    Raises
    ------
    ValueError
    """
    if (
        not isinstance(base, (int, np.integer)) or
        not isinstance(base, (float, np.inexact))
    ):
        raise ValueError('base must be int or float.')

    ans = base * math.ceil((float(x)/base))
    if isinstance(base, (int, np.integer)):
        return int(ans)
    return ans

######################
# CustomTime
######################


class FormatError(Exception):
    """Error class for format

    """
    def __init__(self, message, content=None):
        self.message = message
        self.content = content

        super().__init__(message)


class CTimeNull():
    """Null object -- most interactions returns Null
    """
    _instance = None

    def _null_eq(self, *args, **kwargs):
        """
        Custom equality function that returns True for itself and None, and
        False for everything else
        """
        if args[0] is None:
            return True
        elif isinstance(args[0], CTimeNull):
            return True
        return False

    def _null(self, *args, **kwargs):
        """
        Custom internal function to return None for most behaviors when
        checking the results of additions, substractions, etc.
        """
        return None

    __eq__ = __ne__ = __ge__ = __gt__ = __le__ = __lt__ = _null_eq
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
        """
        Divmod needs to return two anwsers so cannot be included in _null.
        """
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


class Ctime():
    """
    A time object handling the base 60 nature of minutes and seconds and the
    base 24 nature of hours and able to represent time as a pure integer like
    in many traffic count files (ie: 700 stands for 7h00 or 615 for 6h15).

    Note: there is a built-in wrap in behavior at the day, ie: at 23h59:59,
    adding a second returns the time back to 0h00:00 without returning that a
    day threshold was passed
    """

    def __init__(self, time: int = None, time_is_hhmm: bool = False) -> None:
        """
        Creates a time object from a stricly integer writing in the form
        HHMMSS, so a 6 digit time is expected. Since integers do not include
        leading 0s, an entry of XXYY is equal to 00:XX:YY.

        Defaulting the time as a null object makes the class able to be used
        with np.vectorize

        Parameters
        ----------
        time : int, optional
            The time to handle. The default is None.
        time_is_HHMM : bool, optional
            When True, transforms a 4 digit number to add trailing 0s, so
            XXYY becomes XX:YY:00. The default is False.

        Raises
        ------
        TypeError
            Somee other class than an int was provided.

        Returns
        -------
        None.

        """
        if time is None:
            # NB: allowing time=None makes the class able to be used with
            # np.vectorize
            self._time = CTimeNull()
        else:
            if not isinstance(time, int):
                try:
                    time = int(time)
                except TypeError as exc:
                    raise TypeError(
                        'expected int, received ' +
                        str(time.__class__.__name__)
                    ) from exc

            if time_is_hhmm:
                time = time * 100

            if time > 235959:
                time = time % 240000

            self._time = time

    @property
    def hhmm(self):
        """Return time as hour minute format

        Returns
        -------
        int
        """
        return self._time // 100

    def __repr__(self):
        if self._time is CTimeNull():
            return self._time.__repr__()
        return (
            f"<Ctime: {self.hour:02}" +
            f"hrs:{self.minute:02}" +
            f"min:{self.second:02}sec>"
        )

    def __eq__(self, other):
        if self._time is CTimeNull():
            return self._time.__eq__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time == other._time:
                return True
            return False
        except (TypeError, ValueError):
            return False

    def __ne__(self, other):
        if self._time is CTimeNull():
            return self._time.__ne__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        if self.__eq__(other):
            return False
        return True

    def __gt__(self, other):
        if self._time is CTimeNull():
            return self._time.__gt__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time > other._time:
                return True
            return False
        except (TypeError, ValueError):
            return False

    def __ge__(self, other):
        if self._time is CTimeNull():
            return self._time.__ge__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time >= other._time:
                return True
            return False
        except (TypeError, ValueError):
            return False

    def __lt__(self, other):
        if self._time is CTimeNull():
            return self._time.__lt__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time < other._time:
                return True
            return False
        except (TypeError, ValueError):
            return False

    def __le__(self, other):
        if self._time is CTimeNull():
            return self._time.__le__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            if self._time <= other._time:
                return True
            return False
        except (TypeError, ValueError):
            return False

    def __str__(self):
        if self._time is CTimeNull():
            return self._time.__str__()
        return self.as_string()

    def __add__(self, other):
        if self._time is CTimeNull():
            return self._time.__add__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        elif isinstance(other, str):
            pass

        try:
            if other._time < 0:
                return self.__sub__(Ctime(-1*other._time))

            hour1 = self.hour
            hour2 = other.hour
            min1 = self.minute
            min2 = other.minute
            sec1 = self.second
            sec2 = other.second

            add_hour = (
                hour1 + hour2 + (min1 + min2 + (sec1 + sec2) // 60) //
                60
            ) % 24 * 10000
            add_min = (min1 + min2 + (sec1 + sec2) // 60) % 60 * 100
            add_sec = (sec1 + sec2) % 60

            return self.__class__(add_hour + add_min + add_sec)
        except (ValueError, TypeError):
            return NotImplemented

    def __iadd__(self, other):
        if self._time is CTimeNull():
            return self._time.__iadd__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)

        try:
            if other._time < 0:
                return self.__sub__(Ctime(-1*other._time))

            hour1 = self.hour
            hour2 = other.hour
            min1 = self.minute
            min2 = other.minute
            sec1 = self.second
            sec2 = other.second

            add_hour = (
                hour1 + hour2 + (min1 + min2 + (sec1 + sec2) // 60) //
                60
            ) % 24 * 10000
            add_min = (min1 + min2 + (sec1 + sec2) // 60) % 60 * 100
            add_sec = (sec1 + sec2) % 60

            self._time = add_hour + add_min + add_sec

            return self
        except (ValueError, TypeError):
            return NotImplemented

    def __radd__(self, other):
        if self._time is CTimeNull():
            return self._time.__radd__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            return self.__add__(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __sub__(self, other):
        if self._time is CTimeNull():
            return self._time.__sub__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)

        try:
            if other._time < 0:
                return self.__add__(Ctime(-1*other._time))

            hour1 = self.hour
            hour2 = other.hour
            min1 = self.minute
            min2 = other.minute
            sec1 = self.second
            sec2 = other.second

            sub_hour = (
                hour1 - hour2 + (min1 - min2 + (sec1 - sec2) // 60) //
                60
            ) % 24 * 10000
            sub_min = (min1 - min2 + (sec1 - sec2) // 60) % 60 * 100
            sub_sec = (sec1 - sec2) % 60

            return self.__class__(sub_hour + sub_min + sub_sec)
        except (TypeError, ValueError):
            return NotImplemented

    def __rsub__(self, other):
        if self._time is CTimeNull():
            return self._time.__rsub__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            other = Ctime(other)
        try:
            return other.__sub__(self)
        except (TypeError, ValueError):
            return NotImplemented

    def __mul__(self, other):
        if self._time is CTimeNull():
            return self._time.__mul__(other)
        _self_time = self.hour * 3600 + self.minute * 60 + self.second

        if _self_time == 0:    # log 0 will fail
            return self.__class__(0)

        if isinstance(other, numbers.Number) and not isinstance(other, bool):

            if other == 0:     # log 0 will fail
                return self.__class__(0)

            final_seconds = int(round(
                math.exp(
                    math.log(_self_time) + math.log(other)
                ),
                0
            ))
            secs = final_seconds % 60
            mins = (final_seconds // 60) % 60 * 100
            hour = (final_seconds // 60) // 60 * 10000

            return self.__class__(hour+mins+secs)

        try:
            if other._time == 0:   # log 0 will fail
                return self.__class__(0)

            _other_time = other.hour * 3600 + other.minute * 60 + other.second
            return Ctime(
                int(round(
                    math.exp(math.log(_self_time) - math.log(_other_time)),
                    0
                ))
            )
        except (TypeError, ValueError):
            return NotImplemented

    def __rmul__(self, other):
        if self._time is CTimeNull():
            return self._time.__rmul__(other)
        return self.__mul__(other)

    def __truediv__(self, other):
        if self._time is CTimeNull():
            return self._time.__truediv__(other)
        _self_time = self.hour * 3600 + self.minute * 60 + self.second

        if _self_time == 0:   # log 0 will fail
            return self.__class__(0)

        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            # Ctime / number returns a Ctime
            if other == 0:
                raise ZeroDivisionError(
                    f'{self.__class__} division by zero using ' +
                    f'{other.__class__}'
                )

            final_seconds = int(round(
                math.exp(math.log(_self_time) - math.log(other)),
                0
            ))
            secs = final_seconds % 60
            mins = (final_seconds // 60) % 60 * 100
            hour = (final_seconds // 60) // 60 * 10000
            return self.__class__(hour+mins+secs)

        try:
            # Ctime / Ctime returns a float
            if other._time == 0:
                raise ZeroDivisionError(
                    f'{self.__class__} division by zero using ' +
                    f'{other.__class__}'
                )

            _self_time = self.hour * 3600 + self.minute * 60 + self.second
            _other_time = other.hour * 3600 + other.minute * 60 + other.second
            return math.exp(math.log(_self_time) - math.log(_other_time))
        except (ValueError, TypeError):
            return NotImplemented

    def __rtruediv__(self, other):
        if self._time is CTimeNull():
            return self._time.__rtruediv__(other)
        if isinstance(other, numbers.Number) and not isinstance(other, bool):
            return Ctime(int(round(other / float(self._time), 0)))
        else:
            return self.__truediv__(other)

    def add_time(self, time: int) -> None:
        """
        Adds an amount of time to the Ctime._time attribute

        Parameters
        ----------
        time : int
            The time to add.

        Returns
        -------
        None.
        """
        self._time = (self + Ctime(time)).time

    def as_datetime(self, **kwargs) -> datetime.time | datetime.datetime:
        """
        Transform self to a datetime.time or a datetime.datetime object. See
        the documention of the ctime_to_datetime function for more
        informations.

        Returns
        -------
        datetime.time or datetime.datetime

        """
        return ctime_to_datetime(self, **kwargs)

    def as_string(self, **kwargs) -> str:
        """
        Transform self to a string object.

        Parameters
        ----------
        **kwargs :
            Keywords compatible with the ctime_as_string function..

        Returns
        -------
        str
            The timestamp as a string object.
        """
        return ctime_as_string(self, **kwargs)

    @property
    def time(self) -> int:
        """ Return time property

        Returns
        -------
        int
        """
        if self._time is CTimeNull():
            return None
        return self._time

    @property
    def second(self) -> int:
        """ Return time as seconds

        Returns
        -------
        int
        """
        return self._get_sec()

    @property
    def minute(self) -> int:
        """ Return time as minutes

        Returns
        -------
        int
        """
        return self._get_min()

    @property
    def hour(self) -> int:
        """ Return time as hours

        Returns
        -------
        int
        """
        return self._get_hour()

    @time.setter
    def time(self, value: int) -> None:
        self._time = Ctime(value)

    @second.setter
    def second(self, value: int) -> None:
        hhmm = (self._time // 100) * 100
        self._time = (Ctime(hhmm) + Ctime(value)).time

    @minute.setter
    def minute(self, value: int) -> None:
        hh = (self._time // 10000) * 10000
        ss = self._time % 100
        self._time = (Ctime(hh) + Ctime(value * 100) + Ctime(ss)).time

    @hour.setter
    def hour(self, value: int) -> None:
        mmss = self._time % 10000
        self._time = (Ctime(value * 10000) + Ctime(mmss)).time

    def _get_sec(self) -> int:
        """
        Query only the part of the number that relates to seconds

        Returns
        -------
        int
            Amount of seconds as int
        """
        return self._time % 100

    def get_total_sec(self) -> int:
        """
        Transform the timestamp in a number of seconds from 0:00:00.

        Returns
        -------
        int
            The total number of seconds from midnight.

        """
        return self.hour * 3600 + self.minute * 60 + self.second

    def get_total_min(self, add_sec_as_float: bool = False) -> int | float:
        """
        Transform the timestamp in a number of minutes from 0:00:00.

        Parameters
        ----------
        add_sec_as_float : bool, optional
            If true, seconds are returned as a fractional part, the output thus
            becoming a float. The default is False.

        Returns
        -------
        int | float
            The total number of minutes from midnight.
        """
        if not add_sec_as_float:
            return self.hour * 60 + self.minute
        else:
            return self.hour * 60 + self.minute + self.second / 60.0

    def get_total_hour(
        self,
        add_min_sec_as_float: bool = False
    ) -> int | float:
        """
        Transform the timestamp in a number of hours from 0:00:00.

        Parameters
        ----------
        add_min_sec_as_float : bool, optional
            If true, seconds are returned as a fractional part, the output thus
            becoming a float. Note that this part will be returned as a base 10
            number, and thus cannot be directly converted into minutes and
            seconds. The default is False.

        Returns
        -------
        int | float
            The total number of minutes from midnight.
        """
        if not add_min_sec_as_float:
            return self.hour
        return self.hour + self.minute/60.0 + self.second / 3600.0

    def _get_min(self):
        """
        Query only the part of the number that relates to minutes

        Returns
        -------
        int
            Amount of minutes as int
        """
        return (self._time // 100) % 100

    def _get_hour(self):
        """
        Query only the part of the number that relates to hours

        Returns
        -------
        int
            Amount of hours as int
        """
        return self._time // 10000

    def round_time(self, **kwargs) -> int | float:
        """
        Round the timestamp using the round_ctime function.

        Parameters
        ----------
        **kwargs :
            Keywords compatible with the round_ctime function.

        Returns
        -------
        int | float
            The rounded time.
        """
        return round_ctime(self, **kwargs)

    @staticmethod
    def from_string(time: str,
                    hour_format: str | None = None,
                    separator: str = ':',
                    return_hour_format: bool = False
                    ) -> 'Ctime':
        """
        Builds a Ctime object from a str object formatted into standard time
        formats.

        Parameters
        ----------
        time : str
            The string from which to extract the timestamp.
        hour_format : str | None, optional
            Accepted formats are:
                - 24 hour formats:
                    "HH:MM:SS"       example: 13:05:32
                    "HH:MM"          example: 13:05

                - 12 hour formats:
                    "12H HH:MM:SS"   example: 01:05:32 PM
                    "12H HH:MM "     example: 01:05 PM

            If None, the method will try to detect the format, including the
            separators. The default is None.
        separator : str, optional
            The separator(s) used to denote the different parts of the
            timestamp.  Note that the same one needs to be used between the
            hours, minutes and, seconds parts. The default is ':'.
        return_hour_format : bool, optional
            If True, returns the detected format. If the format was giving,
            this will return the value of the hour_format keyword.
            The default is False.

        Returns
        -------
        Ctime
            The Ctime object built on the timestamp.

        """
        # TODO: add mm:ss format

        if hour_format is None:

            # first try to determine the separator
            possible_separators = [':', 'h', ' ']
            # the space is problematic is mixed with a space before
            # the AM/PM tag
            time = multireplace(
                time,
                {item: ':' for item in possible_separators if item != ' '},
                ignore_case=True
            )
            # check to replace the space with a more rubst algorithm
            if 'AM' in time.upper():
                time = (
                    time.upper()
                        .replace('AM', '')
                        .strip()
                        .replace(' ', separator) + ' AM'
                )
            elif 'PM' in time.upper():
                time = (
                    time.upper()
                        .replace('PM', '')
                        .strip()
                        .replace(' ', separator) + ' PM'
                )
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

        time = time.replace(separator, ':')

        if hour_format == 'HH:MM:SS':
            ctime = Ctime(int(time.replace(':', '')))

        elif hour_format == 'HH:MM':
            ctime = Ctime(int(time.replace(':', ''))*100)

        elif hour_format == 'HH':
            ctime = Ctime(int(time.replace(':', ''))*10000)

        elif hour_format == '12H HH:MM:SS':
            if 'AM' in time:
                ctime = Ctime.from_string(
                    time.replace('AM', '').strip(),
                    hour_format='HH:MM:SS'
                )
            if 'PM' in time:
                ctime = Ctime.from_string(
                    time.replace('PM', '').strip(),
                    hour_format='HH:MM:SS'
                ) + Ctime(120000)

        elif hour_format == '12H HH:MM':
            if 'AM' in time.upper():
                ctime = Ctime.from_string(
                    time.replace('AM', '').strip(),
                    hour_format='HH:MM'
                )
                if ctime.time == 120000:
                    ctime = Ctime(0)

            if 'PM' in time.upper():
                ctime = Ctime.from_string(
                    time.replace('PM', '').strip(),
                    hour_format='HH:MM'
                )
                if ctime.time < 120000:
                    ctime = ctime + 120000

        elif hour_format == '12H HH':
            if 'AM' in time.upper():
                ctime = Ctime.from_string(
                    time.replace('AM', '').strip(),
                    hour_format='HH'
                )
                if ctime.time == 120000:
                    ctime = Ctime(0)

            if 'PM' in time.upper():
                ctime = Ctime.from_string(
                    time.replace('PM', '').strip(),
                    hour_format='HH'
                )
                if ctime.time < 120000:
                    ctime = ctime + 120000

        if return_hour_format:
            return ctime, hour_format
        else:
            return ctime

    @staticmethod
    def from_declared_times(
        hours: int | float = 0,
        minutes: int | float = 0,
        seconds: int | float = 0
    ) -> 'Ctime':
        """
        Builds a Ctime objects from a given amount of hours, minutes and
        seconds.

        Parameters
        ----------
        hours : int or float
             The number of hours. If a float is passed, the floating part will
             be added to the minutes as a fraction of 60 minutes (ie: 5.80
             hours = 5 hours and 48 minutes). For the purpose of float
             checking, the `hours` parameter is processed before the `minutes`
             parameter.
             (Default value = 0)

        minutes : int or float
            The number of minutes. If a float is passed, the floating part will
            be added to the seconds as a fraction of 60 seconds (ie: 5.80
            minutes = 5 minutes and 48 seconds). For the purpose of float
            checking, the `minutes` parameter is processed before the `seconds`
            parameter.
            (Default value = 0)

        seconds : int or float
             If a float is passed, the amount is rounded to the nearest
             integer.
             (Default value = 0)

        Returns
        -------
        Ctime
            The Ctime object built using the declared times.
        """
        if hours % 1 > 0:
            minutes += hours % 1 * 60
            hours = hours // 1

        if minutes % 1 > 0:
            seconds += minutes % 1 * 60
            minutes = minutes // 1

        if seconds % 1 > 0:
            seconds = round(seconds, 0)

        _hour = (hours + (minutes + (seconds) // 60) // 60) % 24 * 10000
        _min = (minutes + (seconds) // 60) % 60 * 100
        _sec = (seconds) % 60

        return Ctime(_hour + _min + _sec)

    @staticmethod
    def from_list(
        times: list,
        method: str | list = 'from_string',
        **kwargs
    ) -> list['Ctime']:
        """
        This method accepts a list of inputs and a method to read them. The
        different elements can be constructed using different methods.

        Parameters
        ----------
        times : list
            A list of string or dict (see `method`)

        method : str or array-like
             The method used to read the elements of `times`. Accepted values
             are "from_string", "from_declared_times", and "from_datetime".
             If different methods must be used for the different elements,
             pass those methods in an array-like object of equal lenght to
             times.
             (Default value = from_string)

             For the "from_string" method, the corresponding time must be a
             string. The format can be either left to be automatically detected
             or specified using keywords accepted by the ctime_from_string
             function. If a format is specified, it will be assumed for evey
             item of times. See the ctime_from_string function for more
             details.

             For the "from_declared_times" method, the corresponding time must
             be a dictionnary containing at least one of the following
             keywords: `hours`, `minutes`, or `seconds` and their corresponding
             values.  (ex: {hours=13, minutes=58}, {hours=13, minutes=58,
             seconds=25}).

             For the "from_datetime" method,  the corresponding time must be a
             datetime.datetime object.

        kwargs :
            Keywords compatible with the ctime_from_string function.

        Returns
        -------
        list of Ctime objects
        """

        if isinstance(method, str):
            method = [method for i in range(len(times))]

        out = []
        for time, meth in zip(times, method):
            if meth == 'from_string':
                if 'return_hour_format' in kwargs:
                    kwargs['return_hour_format'] = False
                out.append(Ctime.from_string(time, **kwargs))

            if meth == 'from_declared_times':
                out.append(
                    Ctime.from_declared_times(
                        hours=time['hours'] if 'hours' in time.keys() else 0,
                        minutes=time['minutes'] if 'minutes' in time.keys()
                        else 0,
                        seconds=time['seconds'] if 'seconds' in time.keys()
                        else 0
                    )
                )
            if meth == 'from_datetime':
                out.append(Ctime.from_datetime(time))
        return out

    @staticmethod
    def from_datetime(time: datetime.datetime) -> 'Ctime':
        """
        Builds a Ctime objects from a datetime.datetime object.

        Parameters
        ----------
        time : datetime.datetime
            The datetime object to convert.

        Returns
        -------
        Ctime
            The Ctime object built.

        """
        if isinstance(time, np.datetime64):
            # note: we treat it this way so we can vectorize it to call on a
            # pandas df in vector form
            time = pandas.Timestamp(time)
        return Ctime.from_declared_times(
            hours=time.hour,
            minutes=time.minute,
            seconds=time.second
        )

    def isbetween(
        self,
        ctimes: list['Ctime', 'Ctime'],
        include_bot: bool = False,
        include_top: bool = False
    ) -> bool:
        """
        Verify if a Ctime object is situated between two other Ctime objects.

        Parameters
        ----------
        ctimes : list['Ctime', 'Ctime']
            The other times to use for testing.
        include_bot : bool, optional
            If True, used the ">=" operator while if False, uses the
            ">" operator.
            The default is False.
        include_top bool : TYPE, optional
            If True, used the "<=" operator while if False, uses the
            "<" operator.
            The default is False.

        Returns
        -------
        bool

        """
        if include_bot:
            bot_eq = operator.ge
        else:
            bot_eq = operator.gt

        if include_top:
            top_eq = operator.le
        else:
            top_eq = operator.lt

        return bot_eq(self, min(ctimes)) and top_eq(self, max(ctimes))


def ctime_as_string(
    ctime: Ctime,
    string_format: str = 'hhmmss',
    separate_time_segments: bool = True,
    min_sec_separator: str = ':',
    hour_min_separator: str = ':'
) -> str:
    """
    Transform a Ctime notation to a string object.

    Parameters
    ----------
    ctime : Ctime
        The Ctime object to transform.
    stringFormat : str, optional
        The format to use to return the timestamp. Accepted formats are
        'hhmmss', 'hhmm', 'hmm', 'mmss', gtfs. The default is 'hhmmss'.
    separate_time_segments : bool, optional
        When True, textual separators (eg: ':', 'h', etc.) are used to
        denote the different time portions like hours, minutes, and seconds.
        The default is True.
    min_sec_separator : str, optional
        The separator to use between the minutes and seconds part of the
        timestamp. Only operational if separate_time_segments is True.
        The default is ':'.
    hour_min_separator : str, optional
        The separator to use between the hours and minutes part of the
        timestamp. Only operational if separate_time_segments is True.
        The default is ':'.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    FormatError
        The format provided by the stringFormat keyword is unknown.

    Returns
    -------
    str
        The timestamp as a string object.
    """

    hour, minute, second = ctime.hour, ctime.minute, ctime.second

    if string_format == 'gtfs':
        string_format = 'hhmmss'
        separate_time_segments = True
        min_sec_separator = ':'
        hour_min_separator = ':'

    if string_format == 'hhmmss' and separate_time_segments:
        return f'{hour:02d}{hour_min_separator}{minute:02d}' + \
               f'{min_sec_separator}{second:02d}'
    if string_format == 'hhmmss':
        return f'{hour:02d}{minute:02d}{second:02d}'

    if string_format == 'hhmm' and separate_time_segments:
        return f'{hour:02d}{hour_min_separator}{minute:02d}'
    if string_format == 'hhmm':
        return f'{hour:02d}{minute:02d}'

    if string_format == 'hmm' and separate_time_segments:
        return f'{hour:01d}{hour_min_separator}{minute:02d}'
    if string_format == 'hmm':
        return f'{hour:01d}{minute:02d}'

    if string_format == 'mmss' and separate_time_segments:
        return f'{minute:02d}{min_sec_separator}{second:02d}'
    if string_format == 'mmss':
        return f'{minute:02d}{second:02d}'

    raise FormatError('Unrecognised format')


def int_to_timestring(time: int) -> str:
    """
    Converts an integer to a HHMMSS formatted string, using the french "h"
    separator for the hours and minutes parts and the french ":" for the
    minutes and seconds parts.

    Parameters
    ----------
    time : int
        The integer to convert.

    Returns
    -------
    str
        The timestamp as a sting object.
    """
    return ctime_as_string(
        Ctime(time),
        string_format='hhmmss',
        hour_min_separator='h',
        min_sec_separator=':'
    )


def int_to_gtfs_timestring(time: int) -> str:
    """
    Converts an integer to a GTFS formatted string, using the english ":"
    separator for both hours, minutes and seconds parts.

    Parameters
    ----------
    time : int
        The integer to convert.

    Returns
    -------
    str
        The timestamp as a sting object.
    """
    return ctime_as_string(Ctime(time), string_format='gtfs')


def mean_of_ctimes(ctime_list: list[Ctime]) -> Ctime:
    """
    Calculates the mean time of a list of ctime objects, circumventing the
    overflowing behavior (23h59:59 + 1 ==> 0h00:00) that make a normal sum
    and division combo fail when the sum of individual times make more than
    a full day.

    Parameters
    ----------
    ctime_list : list[Ctime]
        The list of Ctime objects on which to calculate the mean.

    Returns
    -------
    Ctime
        The average time.

    """
    total_seconds = sum(
        item.get_total_sec() for item in ctime_list
        if item.time is not CTimeNull()
    )

    return Ctime.from_declared_times(
        seconds=float(total_seconds)/len(ctime_list)
    )


def signed_ctime_sub(time1: Ctime, time2: Ctime, **kwargs) -> str:
    """
    Since Ctime does not support negative times, this function can be used to
    get a string containing negative timestamps.

    Parameters
    ----------
    time1 : Ctime
        The first time object from which time must be substracted.
    time2 : Ctime
        The second time object, substracted from time1.
    **kwargs :
        Keywords compatible with the ctime_as_string function.

    Returns
    -------
    str
        A string that follows the conversions created by the ctime_as_string
        function.

    Exemple
    -------
         _24h = Ctime(240000)
         _15min = Ctime(1500)
        signed_ctime_sub(_24h, _15min)

        >>> '-23:45:00'

    """
    if time1.time is CTimeNull() or time2.time is CTimeNull():
        return ''
    if time2 <= time1:
        return (time1 - time2).as_string(**kwargs)

    return '-'+(time1 - time2).as_string(**kwargs)


def ctime_to_datetime(
    ctime,
    time_only=True,
    date: str | dict = 'today'
) -> datetime.time | datetime.datetime:
    """
    Transform a Ctime object to a datetime.time or a datetime.datetime object.

    Parameters
    ----------
    ctime : TYPE
        DESCRIPTION.
    time_only : TYPE, optional
        A flag to decide how the transformation should be done. If True, the
        function will return a datetime.time object while when False, a
        datetime.datetime is returned. The default is True.
    date : str | dict, optional
        The day to use when converting to a datetime.datetime object. The
        parameter can either accept 'today' of a dictionnary including the
        following keys: year, month, and day. The default is 'today'.

    Raises
    ------
    TypeError
        The date keyword was passed as something else than 'today' or a
        dictionnary.

    Returns
    -------
    datetime.time or datetime.datetime
        The returned type depends of the time_only keyword's value.

    """
    if ctime.time is CTimeNull():
        return None
    if time_only:
        return datetime.time(
            hour=ctime.hour,
            minute=ctime.minute,
            second=ctime.second
        )

    if date == 'today':
        td = datetime.datetime.today()
        return datetime.datetime(
            year=td.year,
            month=td.month,
            day=td.day,
            hour=ctime.hour,
            minute=ctime.minute,
            second=ctime.second
        )

    if not isinstance(date, dict):
        raise TypeError
    return datetime.datetime(
        year=date['year'],
        month=date['month'],
        day=date['day'],
        hour=ctime.hour,
        minute=ctime.minute,
        second=ctime.second
    )


def round_ctime(
    ctime: Ctime,
    round_unit: str = 'min',
    how: str = 'round',
    round_min_base: int | float | None = None,
    as_string: bool = False,
    str_format_kwargs: dict = None
) -> Ctime | str:
    """
    Round the time contained in a Ctime object according to some rules.

    Parameters
    ----------
    ctime : Ctime
        DESCRIPTION.
    round_unit : str, optional
        The unit where the rounding takes place. Accepted values are 'hour' and
        'min'. The default is 'min'.
    how : str, optional
        Hoe to perform the rounding. Accepted values are 'round', 'ceil', and
        'floor'. The default is 'round'.
    round_min_base : int | float | None, optional
        If an int or float is provided, this number is used to round on. If
        None, the rounding is done on the nearest integer. For exemple,
        assuming 'round_unit' is set to 'min', rounding at the nearest 15
        minutes can be performed by setting round_min_base=15. The default is
        None.
    as_string : bool, optional
        Returns the result as a string using the ctime_as_string function.
        The default is False.
    strFormatKwargs : dict, optional
        Keywords compatible with the ctime_as_string function.

    Returns
    -------
    Ctime | str
        Rounded time either as a Ctime object or as a tring timestamp.

    """
    if not str_format_kwargs:
        str_format_kwargs = {}

    if ctime.time is CTimeNull():
        return None

    if how == 'round':
        func = round_on
    elif how == 'ceil':
        func = ceil_on
    elif how == 'floor':
        func = floor_on

    _hour, _mins, _secs = ctime.hour, ctime.minute, ctime.second

    if round_min_base is not None:
        ctime = Ctime(
            func(func(_secs, 60) // 60 + _mins, round_min_base) * 100
        ) + Ctime(_hour * 10000)

    else:
        if round_unit == 'min':
            ctime = Ctime(
                func(_secs, 60)
            ) + Ctime(_mins * 100) + Ctime(_hour * 10000)

        if round_unit == 'hrs':
            ctime = Ctime(
                func(func(_secs, 60) // 60 + _mins, 60) * 100
            ) + Ctime(_hour * 10000)

    if as_string:
        return ctime.as_string(**str_format_kwargs)

    return ctime


if __name__ == "__main__":
    pass
