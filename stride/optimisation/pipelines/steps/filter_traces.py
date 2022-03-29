
from stride.utils import filters

from ....core import Operator


class FilterTraces(Operator):
    """
    Filter a set of time traces.

    Parameters
    ----------
    f_min : float, optional
        Lower value for the frequency filter, defaults to None (no lower filtering).
    f_max : float, optional
        Upper value for the frequency filter, defaults to None (no upper filtering).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f_min = kwargs.pop('f_min', None)
        self.f_max = kwargs.pop('f_max', None)
        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)

        filtered = []
        for each in traces:
            filtered.append(self._apply(each, **kwargs))

        if len(traces) > 1:
            return tuple(filtered)

        else:
            return filtered[0]

    def adjoint(self, *d_traces, **kwargs):
        d_traces = d_traces[:self._num_traces]

        filtered = []
        for each in d_traces:
            filtered.append(self._apply(each, **kwargs))

        self._num_traces = None

        if len(d_traces) > 1:
            return tuple(filtered)

        else:
            return filtered[0]

    def _apply(self, traces, **kwargs):
        time = traces.time

        f_min = self.f_min*time.step if self.f_min is not None else 0
        f_max = self.f_max*time.step if self.f_max is not None else 0

        out_traces = traces.alike(name='filtered_%s' % traces.name)

        if self.f_min is None and self.f_max is not None:
            # filtered = filters.lowpass_filter_fir(traces.extended_data, f_max)
            filtered = filters.lowpass_filter_butterworth(traces.extended_data, f_max)

        elif self.f_min is not None and self.f_max is None:
            # filtered = filters.highpass_filter_fir(traces.extended_data, f_min)
            filtered = filters.highpass_filter_butterworth(traces.extended_data, f_min)

        elif self.f_min is not None and self.f_max is not None:
            # filtered = filters.bandpass_filter_fir(traces.extended_data, f_min, f_max)
            filtered = filters.bandpass_filter_butterworth(traces.extended_data, f_min, f_max)

        else:
            filtered = traces.extended_data

        out_traces.extended_data[:] = filtered

        return out_traces
