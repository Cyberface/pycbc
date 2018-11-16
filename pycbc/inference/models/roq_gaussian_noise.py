# Copyright (C) 2018  Sebastian Khan
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""This module provides model classes that assume the noise is Gaussian
and uses a reduced order quadrature rule to approximate this.
"""

import numpy

from pycbc import filter as pyfilter
from pycbc.waveform import NoWaveformError
from pycbc.types import Array

from .base_data import BaseDataModel

class ROQGaussianNoise(BaseDataModel):
    r""" fill me in
    fill me in
    fill me in
    fill me in
    fill me in

    """
    name = 'roq_gaussian_noise'

    def __init__(self, data, psd, deltaF, linear_weights, quadratic_weights,
                 linear_freq_nodes, quadratic_freq_nodes, waveform_generator,
                 variable_params, **kwargs):
        """
        linear_weghts and quadratic_weights are dict with det ifo name as keys
        also the *_freq_nodes
        """

        super(ROQGaussianNoise, self).__init__(variable_params, data,
                                            waveform_generator, **kwargs)
        self.psd = psd
        self.deltaF = deltaF
        self.linear_weights = linear_weights
        self.quadratic_weights = quadratic_weights
        self.linear_freq_nodes = linear_freq_nodes
        self.quadratic_freq_nodes = quadratic_freq_nodes

        self.d_dot_d = {}
        for d in self.data:
            tmp = (numpy.vdot(self.data[d],self.data[d]/self.psd[d])*4.*self.deltaF).real
            self.d_dot_d.update({d:tmp})

    @staticmethod
    def BuildROQWeights(data, B, deltaF):
        """Implementation of eq. 9 and 10 of arxiv:1604.08253
        Computes the ROQ weights for a given data stretch

        data: ifo strain data, sample rate and
            number of samples must match the ROM basis matrix resolution.
            i.e. data.shape[0] == B.shape[0]
        B: ROM basis matrix
        deltaF: frequency resolution for integration
        """
        assert (data.shape[0] == B.shape[0]),"data does not match shape of B matrix"

        weights = numpy.dot(data, B.conjugate()) * deltaF * 4.
        return weights

    def _loglr(self):
        """Low-level function that calculates the log likelihood ratio.

        Computes the log likelihood of the paramaters using a
        reduced order quadrature (ROQ) approximation
        """

        params = self.current_params
        # compute waveforms
        # FIXME:
        # something needs to change here because in principle the linear_freq_nodes depend on the ifo... maybe
        # this doesn't take into account the low and high frequency cut off in the likelihood integral
        try:
            wfs_linear = self._waveform_generator.generate(
                sample_points_per_detectors=self.linear_freq_nodes,
                **params)
        except NoWaveformError:
            return self._nowaveform_loglr()

        try:
            wfs_quad = self._waveform_generator.generate(
                sample_points_per_detectors=self.quadratic_freq_nodes,
                **params)
        except NoWaveformError:
            return self._nowaveform_loglr()

        lr = 0
        for det in wfs_linear.keys():

            # observed strain, empirical intpolate at linear frequency nodes
            eim_strain_linear = wfs_linear[det]

            # observed strain, empirical intpolate at quadratic frequency nodes
            h_quad = wfs_quad[det]
            # appropriate combination for <h,h> term in likelihood
            eim_strain_quad = h_quad.numpy().conjugate() * h_quad.numpy()

            # construct roq likelihood
            #FIXME: not sure about tranpose
            #d_dot_h = (numpy.vdot(eim_strain_linear, self.linear_weights[det])).real
            d_dot_h = (numpy.vdot(eim_strain_linear, self.linear_weights[det].T)).real

            #FIXME:not sure about tranpose
            # h_dot_h = (numpy.vdot(eim_strain_quad, self.quadratic_weights[det].T)).real
            h_dot_h = (numpy.vdot(eim_strain_quad, self.quadratic_weights[det])).real

            # equation 11 arxiv:1604.08253
            lr += 0.5 * ( -2*d_dot_h + h_dot_h - self.d_dot_d[det] )

        self._current_stats.loglikelihood = lr + self.lognl

        return float(lr)

    def _lognl(self):
        """Computes the log likelihood assuming the data is noise.

        Since this is a constant for Gaussian noise, this is only computed once
        then stored.
        """

        # FIXME: not implemented yet - get this from gaussian_noise.py

        # this is the self.d_dot_d dictionary for me

        return 1.

    def _loglikelihood(self):
        r"""Computes the log likelihood of the paramaters,

        # FIXME: add correct doc string

        Returns
        -------
        float
            The value of the log likelihood evaluated at the given point.
        """
        # since the loglr has fewer terms, we'll call that, then just add
        # back the noise term that canceled in the log likelihood ratio
        return self.loglr + self.lognl