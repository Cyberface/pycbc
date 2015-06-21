#! /usr/bin/env python

import os, sys
import math
import functools
import operator
import sqlite3
import numpy
import lal
import matplotlib
from matplotlib import pyplot

from pycbc import distributions

from glue import segments
from pylal import printutils

def drop_trailing_zeros(num):
    """
    Drops the trailing zeros in a float that is printed.
    """
    txt = '%f' %(num)
    txt = txt.rstrip('0')
    if txt.endswith('.'):
        txt = txt[:-1]
    return txt

def get_signum(val, err, max_sig=numpy.inf):
    """
    Given an error, returns a string for val formated to the appropriate
    number of significant figures.
    """
    coeff, pwr = ('%e' % err).split('e')
    if pwr.startswith('-'):
        pwr = int(pwr[1:])
        if round(float(coeff)) == 10.:
            pwr -= 1
        pwr = min(pwr, max_sig)
        tmplt = '%.' + str(pwr) + 'f'
        return tmplt % val
    else:
        pwr = int(pwr[1:])
        if round(float(coeff)) == 10.:
            pwr += 1
        # if the error is large, we can sometimes get 0;
        # adjust the round until we don't get 0 (assuming the actual
        # value isn't 0)
        return_val = round(val, -pwr)
        if val != 0.:
            loop_count = 0
            max_recursion = 100
            while return_val == 0.:
                pwr -= 1
                return_val = round(val, -pwr)
                loop_count += 1
                if loop_count > max_recursion:
                    raise ValueError("Maximum recursion depth hit! Input " +\
                        "values are: val = %f, err = %f" %(val, err))
        return drop_trailing_zeros(return_val)

def get_color_grayscale(clr):
    """
    Converts the given color to grayscale. The equation used to do the
    conversion comes from:
    en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

    Parameters
    ----------
    clr: tuple
        A color tuple. Should be in the form of (R, G, B, A).

    Returns
    -------
    grayscale: float
        A number between 0. and 1. representing the grayscale, with 0 = black,
        1 = white.
    """
    r, g, b, a = clr
    return 0.299*r + 0.587*g + 0.114*b

def ColorBarLog10Formatter(y, pos):
    """
    Formats y=log10(x) values for a colorbar so that they appear as 10^y.
    """
    return "$10^{%.1f}$" % y

def create_bounded_colorbar_formatter(minlim, maxlim, formatter=None):
    """
    Creates a frozen wrapper around the given colorbar formatter function
    that adds a < (>) symbol to the ticklabel if the y value is <= (>=) minlim
    (maxlim).
    """
    if formatter is None:
        # FIXME: I don't know how colorbar generates it's default formatter,
        # so I'm just creating a dummy colorbar to get it
        dumbfig = pyplot.figure()
        dumbax = dumbfig.add_subplot(111)
        dumbsc = dumbax.scatter(range(2), range(2), c=(0.9*minlim, 1.1*maxlim),
            vmin=minlim, vmax=maxlim)
        dumbcb = dumbfig.colorbar(dumbsc)
        formatter = dumbcb.formatter
        del dumbfig, dumbax, dumbsc, dumbcb
    def bounded_colorbar_formatter(y, pos, formatter, minlim, maxlim):
        tickstr = formatter(y, pos)
        if tickstr.startswith('$'):
            tickstr = tickstr[1:]
            add_dollar = '$'
        else:
            add_dollar = ''
        if y <= minlim:
            tickstr = '< %s' %(tickstr)
        if y >= maxlim:
            tickstr = '> %s' %(tickstr)
        return add_dollar + tickstr
    return pyplot.FuncFormatter(functools.partial(
        bounded_colorbar_formatter, formatter=formatter, minlim=minlim,
        maxlim=maxlim))

def empty_plot(ax, message="Nothing to plot"):
    """
    Creates an empty plot on the given axis.
    """
    # ensure the axis background is white, so the text can be read
    ax.set_axis_bgcolor('w')
    return ax.annotate(message, (0.5, 0.5))

def empty_hatched_plot(ax, hatch="x"):
    """
    Creates an empty plot on the given axis consisting entirely of a hatched
    background.
    """
    tile = ax.fill((0,1,1,0), (0,0,1,1), hatch=hatch, facecolor='w')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    return tile


#############################################
#
#   Data Storage and Slicing tools
#
#############################################

class Template(object):
    """
    Class to store information about a template for plotting.
    """
    # we'll group the various parameters by type
    _intrinsic_params = [
        'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
        'spin2x', 'spin2y', 'spin2z', 'eccentricity',
        'lambda1', 'lambda2'
        ]
    _extrinsic_params = [
        'phi0', 'inclination', 'distance'
        ]
    _waveform_params = [
        'sample_rate', 'segment_length', 'duration',
        'f_min', 'f_ref', 'f_max',
        'axis_choice', 'modes_flag',
        'amp_order', 'phase_order', 'spin_order', 'tidal_order',
        'approximant', 'taper'
        ]
    _ifo_params = [
        'ifo', 'sigma'
        ]
    _tmplt_weight_params = ['weight', 'weight_function']
    _id_name = 'tmplt_id'
    __slots__ = [_id_name] + _intrinsic_params + _extrinsic_params + \
        _waveform_params + _ifo_params + _tmplt_weight_params

    def __init__(self, **kwargs):
        default = None
        [setattr(self, param, kwargs.pop(param, default)) for param in \
            self.__slots__]
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))

    # some other derived parameters
    @property
    def mtotal(self):
        return self.mass1 + self.mass2

    @property
    def mtotal_s(self):
        return lal.MTSUN_SI*self.mtotal

    @property
    def q(self):
        return self.mass1 / self.mass2

    @property
    def eta(self):
        return self.mass1*self.mass2 / self.mtotal**2.

    @property
    def mchirp(self):
        return self.eta**(3./5)*self.mtotal

    @property
    def chi(self):
        return (self.mass1*self.spin1z + self.mass2*self.spin2z) / self.mtotal

    # some short cuts
    @property
    def m1(self):
        return self.mass1

    @property
    def m2(self):
        return self.mass2

    def tau0(self, f0=None):
        """
        Returns tau0. If f0 is not specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (5./(256 * numpy.pi * f0 * self.eta)) * \
            (numpy.pi * self.mtotal_s * f0)**(-5./3.)
   
    def v0(self, f0=None):
        """
        Returns the velocity at f0, as a fraction of c. If f0 is not
        specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (2*numpy.pi* f0 * self.mtotal_s)**(1./3)

    @property
    def s1(self):
        return numpy.array([self.spin1x, self.spin1y, self.spin1z])

    @property
    def s1x(self):
        return self.spin1x

    @property
    def s1y(self):
        return self.spin1y

    @property
    def s1z(self):
        return self.spin1z

    @property
    def s2(self):
        return numpy.array([self.spin2x, self.spin2y, self.spin2z])

    @property
    def s2x(self):
        return self.spin2x

    @property
    def s2y(self):
        return self.spin2y

    @property
    def s2z(self):
        return self.spin2z

    @property
    def apprx(self):
        return self.approximant


class SnglIFOInjectionParams(object):
    """
    Class that stores information about an injeciton in a specific detector.
    """
    __slots__ = ['ifo', 'end_time', 'end_time_ns', 'sigma']
    def __init__(self, **kwargs):
        [setattr(self, param, kwargs.pop(param, None)) for param in \
            self.__slots__]
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))



class Injection(Template):
    """
    Class to store information about an injection for plotting. Inherits from
    Template, and adds slots for sky location.
    """
    # add information about location and time in geocentric coordinates, and
    # the distribution from which the injections were drawn
    _inj_params = [
        'geocent_end_time', 'geocent_end_time_ns', 'ra', 'dec', 'astro_prior',
        'min_vol', 'vol_weight', 'mass_distr', 'spin_distr'
        ]
    # we'll override some of the parameters in TemplateResult
    _id_name = 'simulation_id'
    # sngl_ifos will be a dictionary pointing to instances of
    # SnglIFOInjectionParams
    _ifo_params = ['sngl_ifos']
    __slots__ = [_id_name] + Template._intrinsic_params + \
        Template._extrinsic_params + Template._waveform_params + \
        _ifo_params + _inj_params
    def __init__(self, **kwargs):
        # ensure sngl_ifos is a dictionary
        self.sngl_ifos = kwargs.pop('sngl_ifos', {})
        # set the default for the rest to None
        [setattr(self, param, kwargs.pop(param, None)) for param in \
            self.__slots__ if param != 'sngl_ifos']
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))


class Result(object):
    """
    Class to store a template with an injection, along with event information
    (ranking stat, etc.) for purposes of plotting.

    Information about the template and the injection (masses, spins, etc.)
    are stored as instances of Template and Injection; since these contain
    static slots, this saves on memory if not all information is needed.

    Information about a trigger (snr, chisq, etc.) are stored as attributes
    of the class. Since the class has no __slots__, additional statistics
    may be added on the fly.

    The class can also be used to only store information about a trigger;
    i.e., adding an injection is not necessary.
    
    To make accessing intrinsic parameters easier, set the psuedoattr_class;
    this will make the __slots__ of either the injection or the template
    attributes of this class. See set_psuedoattr_class for details. 
    """
    _psuedoattr_class = None

    def __init__(self, unique_id=None, database=None, event_id=None,
            tmplt=None, injection=None):
        self.unique_id = None
        self.database = None
        self.event_id = None
        if tmplt is None:
            self.template = Template()
        else:
            self.template = tmplt
        if injection is None:
            self.injection = Injection()
        else:
            self.injection = injection
        self._psuedoattr_class = None
        self.snr = None
        self.chisq = None
        self.chisq_dof = None
        self.new_snr = None
        self.false_alarm_rate = None
        self.uncombined_far = None
        self.false_alarm_probability = None
        # experiment parameters
        self.instruments_on = None
        self.livetime = None
        # banksim parameters
        self.effectualness = None
        self.snr_std = None
        self.chisq_std = None
        self.new_snr_std = None
        self.num_samples = None

    def __getattr__(self, name):
        """
        This will get called if __getattribute__ fails. Thus, we can use
        this to access attributes of the psuedoattr_class, if it is set.
        """
        try:
            return object.__getattribute__(self,
                '_psuedoattr_class').__getattribute__(name)
        except AttributeError:
            raise AttributeError("'Result' object has no attribute '%s'" %(
                name))

    def __setattr__(self, name, value):
        """
        First tries to set the attribute in self. If name is not in self's
        dict, next tries to set the attribute in self._psuedoattr_class.
        If that fails with an AttributeError, it then adds the name to self's
        namespace with the associated value.
        """
        try:
            object.__getattribute__(self,
                '_psuedoattr_class').__setattr__(name, value)
        except AttributeError:
            object.__setattr__(self, name, value)

    @property
    def psuedoattr_class(self):
        return self._psuedoattr_class

    def set_psuedoattr_class(self, psuedo_class):
        """
        Makes the __slots__ of the given class visible to self's namespace.
        An error is raised if self and psuedo_class have any attributes
        that have the same name, as this can lead to unexpected behavior.

        Parameters
        ----------
        psuedo_class: {self.template|self.injection}
            An instance of a class to make visible. Should be either self's
            injection or template (but can be any instance of any class).
        """
        # check that there is no overlap
        attribute_overlap = [name for name in self.__dict__ \
            if name in psuedo_class.__slots__]
        if attribute_overlap != []:
            raise AttributeError(
                "attributes %s " %(', '.join(attribute_overlap)) + \
                "are common to self and the given psuedo_class. Delete " +\
                "these attributes from self if you wish to use the given " +\
                "psuedo_class.")
        self._psuedoattr_class = psuedo_class
        
    @property
    def optimal_snr(self):
        """
        Returns the quadrature sum of the inj_sigmas divided by the distance.
        """
        return numpy.sqrt((numpy.array([ifo.sigma \
            for ifo in self.injection.sngl_ifos])**2.).sum()) / \
            self.injection.distance

    # some short cuts
    @property
    def tmplt(self):
        return self.template

    @property
    def inj(self):
        return self.injection


# FIXME: dataUtils in pylal should be moved to pycbc, and the get_val in there
# used instead
def get_arg(row, arg):
    """
    Retrieves an arbitrary argument from the given row object. For speed, the
    argument will first try to be retrieved using getattr. If this fails (this
    can happen if a function of several attributes of row are requested),
    then Python's eval command is used to retrieve the argument. The argument
    can be any attribute of row, or functions of attributes of the row
    (assuming the relevant attributes are floats or ints). Allowed functions
    are anything in Python's math library. No other functions (including
    Python builtins) are allowed.

    Parameters
    ----------
    row: any instance of a Python class
        The object from which to apply the given argument to.
    arg: string
        The argument to apply.

    Returns
    -------
    value: unknown type
        The result of evaluating arg on row. The type of the returned value
        is whatever the type of the data element being retreived is.
    """
    try:
        return getattr(row, arg)
    except AttributeError:
        row_dict = dict([ [name, getattr(row,name)] for name in dir(row)])
        safe_dict = dict([ [name,val] for name,val in \
            row_dict.items()+math.__dict__.items() \
            if not name.startswith('__')])
        return eval(arg, {"__builtins__":None}, safe_dict)


def result_in_range(result, test_dict):
    cutvals = [(get_arg(result, criteria), low, high) \
        for criteria,(low,high) in test_dict.items()] 
    return not any(x < low or x >= high for (x, low, high) in cutvals)


def result_is_match(result, test_dict):
    try:
        matchvals = [(getattr(result, criteria), targetval) 
            for criteria,targetval in test_dict.items()]
    except:
        matchvals = [(get_arg(result, criteria), targetval) \
            for criteria,targetval in test_dict.items()]
    return not any(x != targetval for (x, tagetval) in matchvals)


def apply_cut(results, test_dict):
    return [x for x in results if result_in_range(x, test_dict)]

def slice_results(results, test_dict):
    return apply_cut(results, test_dict)



#############################################
#
#   Tools to load a results into memory
#
#############################################

# the known result file types that we can parse
known_result_types = ['overlaps', 'pycbc_sqlite']

def parse_results_cache(cache_file):
    filenames = []
    f = open(cache_file, 'r')
    for line in f:
        thisfile = line.split('\n')[0]
        if os.path.exists(thisfile):
            filenames.append(thisfile)
    f.close()
    return filenames

def get_injection_results(filenames, load_inj_distribution=False,
        weight_function='uniform', result_table_name='overlap_results',
        ifo=None, verbose=False):
    sqlquery = """
        SELECT
            sim.process_id, sim.waveform, sim.simulation_id,
            sim.mass1, sim.mass2, sim.spin1x, sim.spin1y, sim.spin1z,
            sim.spin2x, sim.spin2y, sim.spin2z,
            sim.distance, sim.inclination, sim.alpha1, sim.alpha2,
            tmplt.event_id, tmplt.mass1, tmplt.mass2,
            tmplt.spin1x, tmplt.spin1y, tmplt.spin1z,
            tmplt.spin2x, tmplt.spin2y, tmplt.spin2z,
            res.effectualness, res.snr, res.snr_std,
            tw.weight, res.chisq, res.chisq_std, res.chisq_dof, res.new_snr,
            res.new_snr_std, res.num_successes, res.sample_rate,
            res.coinc_event_id
        FROM
            %s AS res""" %(result_table_name) + """
        JOIN
            sim_inspiral as sim, coinc_event_map as map
        ON
            sim.simulation_id == map.event_id AND
            map.coinc_event_id == res.coinc_event_id
        JOIN
            sngl_inspiral AS tmplt, coinc_event_map AS mapB
        ON
            mapB.coinc_event_id == map.coinc_event_id AND
            mapB.event_id == tmplt.event_id
        JOIN
            coinc_definer AS cdef, coinc_event AS cev
        ON
            cev.coinc_event_id == res.coinc_event_id AND
            cdef.coinc_def_id == cev.coinc_def_id
        JOIN
            tmplt_weights as tw
        ON
            tw.tmplt_id == tmplt.event_id AND
            tw.weight_function == cdef.description
        WHERE
            cdef.description == ?
    """
    if ifo is not None:
        sqlquery += 'AND res.ifo == ?'
        get_args = (weight_function, ifo)
    else:
        get_args = (weight_function,)
    results = []
    id_map = {}
    idx = 0
    inj_dists = {}
    for ii,thisfile in enumerate(filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(filenames)),
            sys.stdout.flush()
        if not thisfile.endswith('.sqlite'):
            continue
        connection = sqlite3.connect(thisfile)
        cursor = connection.cursor()
        #try:
        if True:
            for (sim_proc_id, apprx, sim_id, m1, m2, s1x, s1y, s1z,
                    s2x, s2y, s2z, dist, inc, min_vol, inj_weight,
                    tmplt_evid, tmplt_m1, tmplt_m2,
                    tmplt_s1x, tmplt_s1y, tmplt_s1z,
                    tmplt_s2x, tmplt_s2y, tmplt_s2z, ff, snr,
                    snr_std, weight, chisq, chisq_std, chisq_dof, new_snr,
                    new_snr_std, nsamp, sample_rate, ceid) in \
                    cursor.execute(sqlquery, get_args):
                thisRes = Result()
                # id information
                thisRes.unique_id = idx
                thisRes.database = thisfile 
                thisRes.event_id = ceid
                id_map[thisfile, sim_id] = idx
                idx += 1
                # Set the injection parameters: we'll make the injection
                # the psuedo class so we can access its attributes directly
                thisRes.set_psuedoattr_class(thisRes.injection)
                thisRes.injection.simulation_id = sim_id
                thisRes.injection.approximant = apprx
                # ensure that m1 is always > m2
                if m2 > m1:
                    thisRes.injection.mass1 = m2
                    thisRes.injection.mass2 = m1
                    thisRes.injection.spin1x = s2x
                    thisRes.injection.spin1y = s2y
                    thisRes.injection.spin1z = s2z
                    thisRes.injection.spin2x = s1x
                    thisRes.injection.spin2y = s1y
                    thisRes.injection.spin2z = s1z
                else:
                    thisRes.injection.mass1 = m1
                    thisRes.injection.mass2 = m2
                    thisRes.injection.spin1x = s1x
                    thisRes.injection.spin1y = s1y
                    thisRes.injection.spin1z = s1z
                    thisRes.injection.spin2x = s2x
                    thisRes.injection.spin2y = s2y
                    thisRes.injection.spin2z = s2z
                thisRes.injection.distance = dist
                thisRes.injection.inclination = inc
                thisRes.injection.min_vol = min_vol
                thisRes.injection.vol_weight = inj_weight
                thisRes.injection.sample_rate = sample_rate
                # set the template parameters
                thisRes.template.tmplt_id = tmplt_evid
                thisRes.template.mass1 = tmplt_m1
                thisRes.template.mass2 = tmplt_m2
                thisRes.template.spin1x = tmplt_s1x
                thisRes.template.spin1y = tmplt_s1y
                thisRes.template.spin1z = tmplt_s1z
                thisRes.template.spin2x = tmplt_s2x
                thisRes.template.spin2y = tmplt_s2y
                thisRes.template.spin2z = tmplt_s2z
                thisRes.template.weight_function = weight_function
                thisRes.template.weight = weight
                # statistics
                thisRes.effectualness = ff
                thisRes.snr = snr
                thisRes.snr_std = snr_std
                thisRes.chisq = chisq
                thisRes.chisq_sts = chisq_std
                thisRes.chisq_dof = chisq_dof
                thisRes.new_snr = new_snr
                thisRes.new_snr_std = new_snr_std
                thisRes.num_samples = nsamp
                # get the injection distribution information
                if load_inj_distribution:
                    try:
                        thisRes.injection.mass_distr = inj_dists[thisfile,
                            sim_proc_id]
                    except KeyError:
                        # need to load the distribution
                        inj_dists[thisfile, sim_proc_id] = \
                            distributions.get_inspinj_distribution(connection,
                            sim_proc_id)
                        thisRes.injection.mass_distr = inj_dists[thisfile,
                            sim_proc_id]
                results.append(thisRes)

            # try to get the sim_inspiral_params table
            tables = cursor.execute(
                'SELECT name FROM sqlite_master WHERE type == "table"'
                ).fetchall()
            if ('sim_inspiral_params',) in tables:
                # older codes stored the minimum volume and injection weight in
                # the sim_inspiral_params table. If we find that column, we'll
                # get the min_vol and vol_weight from there. Otherwise, the
                # min_vol and the vol_weight are set above. 
                column_names = [name[1] for name in cursor.execute(
                    "PRAGMA table_info(sim_inspiral_params)").fetchall()]
                if "min_vol" in column_names and "weight" in column_names:
                    sipquery = """
                        SELECT
                            sip.simulation_id, sip.ifo, sip.sigmasq,
                            sip.min_vol, sip.weight
                        FROM
                            sim_inspiral_params AS sip
                        """
                else:
                    sipquery = """
                        SELECT
                            sip.simulation_id, sip.ifo, sip.sigmasq,
                            NULL, NULL
                        FROM
                            sim_inspiral_params AS sip
                        """
                for simid, ifo, sigmasq, min_vol, vol_weight in cursor.execute(
                        sipquery):
                    try:
                        thisRes = results[id_map[thisfile, simid]]
                    except KeyError:
                        continue
                    sngl_params = SnglIFOInjectionParams(ifo=ifo,
                        sigma=numpy.sqrt(sigmasq))
                    thisRes.injection.sngl_ifos[ifo] = sngl_params
                    if min_vol is not None:
                        thisRes.injection.min_vol = min_vol
                    if inj_weight is not None:
                        thisRes.injection.vol_weight = vol_weight

        #except sqlite3.OperationalError:
        else:
            cursor.close()
            connection.close()
            continue
        #except sqlite3.DatabaseError:
            cursor.close()
            connection.close()
            print "Database Error: %s" % thisfile
            continue

        connection.close()

    if verbose:
        print >> sys.stdout, ""

    return results, id_map


def get_r_distribution_from_inspinj(connection):
    """
    Gets the distance distribution that was given to inspinj from the
    process_params table.
    """
    sqlquery = """
        SELECT
            process_id, param, value
        FROM
            process_params
        WHERE
            program == "inspinj" AND
            (
                param == "--min-distance" OR
                param == "--max-distance" OR
                param == "--d-distr" OR
                param == "--dchirp-distr"
            )
        """
    rdistrs = {}
    for process_id, param, value in connection.cursor().execute(sqlquery):
        # order of storing things is type, distribution, min, max
        rdistrs.setdefault(process_id, ['', '', 0., 0.])
        if param == "--d-distr":
            rdistrs[process_id][0] = 'distance'
            rdistrs[process_id][1] = value
        if param == "--dchirp-distr":
            rdistrs[process_id][0] = "chirp_dist"
            rdistrs[process_id][1] = "uniform"
        if param == "--min-distance":
            rdistrs[process_id][2] = float(value)/1000. # convert kpc to Mpc
        if param == "--max-distance":
            rdistrs[process_id][3] = float(value)/1000.
    return rdistrs

def cull_injection_results(results, primary_arg='false_alarm_rate',
        primary_rank_by='max', secondary_arg='new_snr',
        secondary_rank_by='min'):
    """
    Given a list of injection results in which the injections are mapped to
    multiple singles, picks the more significant one based on the primary_arg.
    If the two events have the same value in the primary arg, the secondary
    arg is used.
    """
    # get the correct operator to use
    if not (primary_rank_by == 'max' or primary_rank_by == 'min'):
        raise ValueError("unrecognized primary_rank_by %s; " %(
            primary_rank_by) + 'options are "max" or "min"')
    if not (secondary_rank_by == 'max' or secondary_rank_by == 'min'):
        raise ValueError("unrecognized secondary_rank_by %s; " %(
            secondary_rank_by) + 'options are "max" or "min"')
    # find the repeated entries
    sorted_results = sorted(results,
        key=lambda x: int(x.simulation_id.split(':')[-1]))
    id_map = {}
    duplicates = {}
    this_count = 1
    for ii,this_result in enumerate(sorted_results):
        if ii+1 < len(sorted_results) and \
                sorted_results[ii+1].simulation_id == \
                this_result.simulation_id:
            this_count += 1
        elif this_count > 1:
            # pick out the loudest out of the repeated values
            this_group = sorted_results[ii-(this_count-1):ii+1]
            primaries = numpy.array([get_arg(x, primary_arg) \
                for x in this_group])
            if primary_rank_by == 'min':
                keep_idx = numpy.where(primaries == primaries.min())[0]
            else:
                keep_idx = numpy.where(primaries == primaries.max())[0]
            if len(keep_idx) > 1:
                secondaries = numpy.array([
                    get_arg(this_group[jj], secondary_arg) for jj in keep_idx])
                # note: this will just keep the first event if the secondaries
                # are equal
                if secondary_rank_by == 'min':
                    secondary_idx = secondaries.argmin()
                else:
                    secondary_idx = secondaries.argmax()
                keep_idx = keep_idx[secondary_idx]
            else:
                keep_idx = keep_idx[0]
            # set this_result to the desired one; also set this_count = 0 for
            # the next group
            this_result = this_group[keep_idx]
            this_count = 1
        if this_count == 1:
            id_map[this_result.database, this_result.simulation_id] = \
                this_result
    return id_map.values(), id_map

    
# FIXME: add template info correctly
def get_pycbc_sqlite_injection_results(filenames, map_label,
        include_missed_injections=True, load_inj_distribution=True,
        load_vol_weights_from_inspinj=True,
        cull_primary_arg='false_alarm_rate', cull_primary_rank_by='min',
        cull_secondary_arg='new_snr', cull_secondary_rank_by='max',
        verbose=False):
    sqlquery = """
        SELECT
            sim.process_id, sim.waveform, sim.simulation_id,
            sim.mass1, sim.mass2, sim.spin1x, sim.spin1y, sim.spin1z,
            sim.spin2x, sim.spin2y, sim.spin2z,
            sim.distance, sim.inclination, sim.alpha1, sim.alpha2,
            tmplt.event_id, tmplt.mass1, tmplt.mass2,
            tmplt.spin1x, tmplt.spin1y, tmplt.spin1z,
            tmplt.spin2x, tmplt.spin2y, tmplt.spin2z,
            res.false_alarm_rate, res.combined_far, res.snr,
            res.ifos, res.coinc_event_id,
            -- experiment information
            experiment.instruments, exsumm.duration
        FROM
            coinc_inspiral AS res
        JOIN
            sngl_inspiral AS tmplt, coinc_event_map AS mapA
        ON
            mapA.coinc_event_id == res.coinc_event_id AND
            mapA.event_id == tmplt.event_id AND
            mapA.table_name == "sngl_inspiral"
            -- FIXME!
            AND tmplt.ifo == "H1"
        JOIN
            sim_inspiral AS sim, coinc_event_map AS mapB,
            coinc_event_map AS mapC
        ON
            sim.simulation_id == mapB.event_id AND
            mapB.coinc_event_id == mapC.coinc_event_id AND
            mapC.event_id == res.coinc_event_id
        JOIN
            coinc_definer AS cdef, coinc_event AS cev
        ON
            cdef.coinc_def_id == cev.coinc_def_id AND
            cev.coinc_event_id == mapB.coinc_event_id
        JOIN
            experiment, experiment_summary AS exsumm, experiment_map AS exmap
        ON
            exmap.coinc_event_id == res.coinc_event_id AND
            exmap.experiment_summ_id == exsumm.experiment_summ_id AND
            exsumm.experiment_id == experiment.experiment_id AND
            exsumm.datatype == "simulation" AND
            exsumm.sim_proc_id == sim.process_id
        WHERE
            cdef.description == ?
    """
    results = []
    idx = 0
    inj_dists = {}
    for ii,thisfile in enumerate(filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(filenames)),
            sys.stdout.flush()
        if not thisfile.endswith('.sqlite'):
            continue
        connection = sqlite3.connect(thisfile)
        cursor = connection.cursor()
        # if getting the distance distributions from inspinj, get it now
        if load_vol_weights_from_inspinj:
            rdistrs = get_r_distribution_from_inspinj(connection)
        for (sim_proc_id, apprx, sim_id, m1, m2, s1x, s1y, s1z,
                s2x, s2y, s2z, dist, inc, min_vol, vol_weight,
                tmplt_evid, tmplt_m1, tmplt_m2,
                tmplt_s1x, tmplt_s1y, tmplt_s1z,
                tmplt_s2x, tmplt_s2y, tmplt_s2z,
                uncombined_far, combined_far, snr, ifos, ceid,
                ifos_on, livetime) in \
                cursor.execute(sqlquery, (map_label,)):
            thisRes = Result()
            # id information
            thisRes.unique_id = idx
            thisRes.database = thisfile 
            thisRes.event_id = ceid
            idx += 1
            # Set the injection parameters: we'll make the injection
            # the psuedo class so we can access its attributes directly
            thisRes.set_psuedoattr_class(thisRes.injection)
            thisRes.injection.simulation_id = sim_id
            thisRes.injection.approximant = apprx
            # ensure that m1 is always > m2
            if m2 > m1:
                thisRes.injection.mass1 = m2
                thisRes.injection.mass2 = m1
                thisRes.injection.spin1x = s2x
                thisRes.injection.spin1y = s2y
                thisRes.injection.spin1z = s2z
                thisRes.injection.spin2x = s1x
                thisRes.injection.spin2y = s1y
                thisRes.injection.spin2z = s1z
            else:
                thisRes.injection.mass1 = m1
                thisRes.injection.mass2 = m2
                thisRes.injection.spin1x = s1x
                thisRes.injection.spin1y = s1y
                thisRes.injection.spin1z = s1z
                thisRes.injection.spin2x = s2x
                thisRes.injection.spin2y = s2y
                thisRes.injection.spin2z = s2z
            thisRes.injection.distance = dist
            thisRes.injection.inclination = inc
            if load_vol_weights_from_inspinj:
                # get the distribution that was used by inspinj
                r = dist 
                distr_type, distribution, r1, r2 = rdistrs[sim_proc_id]
                # FIXME: the scale_fac and weights are copied from 
                # randr_by_snr; these functions should be moved to library
                # location instead
                if distr_type == "chirp_dist":
                    # scale r1 and r2 by the chirp mass
                    scale_fac = (thisRes.injection.mchirp/\
                        (2.8 * 0.25**0.6))**(5./6)
                    r1 = r1*scale_fac
                    r2 = r2*scale_fac
                if distribution == "volume":
                    min_vol = (4./3)*numpy.pi*r1**3.
                    vol_weight = (4./3)*numpy.pi*(r2**3. - r1**3.)
                elif distribution == "uniform":
                    min_vol = (4./3)*numpy.pi*r1**3.
                    vol_weight = 4.*numpy.pi*(r2-r1) * r**2. 
                elif distribution == "log10":
                    min_vol = (4./3)*numpy.pi*r1**3.
                    vol_weight = 4.*numpy.pi * r**3. * numpy.log(r2/r1)
                else:
                    raise ValueError("unrecognized distribution %s" %(
                        distribution))
            # note: if not loading weights from inspinj, the alpha1 and
            # alpha2 columns will be used
            thisRes.injection.min_vol = min_vol
            thisRes.injection.vol_weight = vol_weight
            # set the template parameters
            thisRes.template.tmplt_id = tmplt_evid
            thisRes.template.mass1 = tmplt_m1
            thisRes.template.mass2 = tmplt_m2
            thisRes.template.spin1x = tmplt_s1x
            thisRes.template.spin1y = tmplt_s1y
            thisRes.template.spin1z = tmplt_s1z
            thisRes.template.spin2x = tmplt_s2x
            thisRes.template.spin2y = tmplt_s2y
            thisRes.template.spin2z = tmplt_s2z
            # statistics
            thisRes.new_snr = snr
            thisRes.uncombined_far = uncombined_far
            thisRes.false_alarm_rate = combined_far
            # experiment
            thisRes.instruments_on = ifos_on
            # convert livetime from seconds to years
            thisRes.livetime = livetime / lal.YRJUL_SI
            # get the injection distribution information
            if load_inj_distribution:
                try:
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        sim_proc_id]
                except KeyError:
                    # need to load the distribution
                    inj_dists[thisfile, sim_proc_id] = \
                        distributions.get_inspinj_distribution(connection,
                        sim_proc_id)
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        sim_proc_id]
            results.append(thisRes)

        # add the outright missed injections
        # we'll get the missed injections using printmissed
        missed_injections = printutils.printmissed(connection,
            'sim_inspiral', 'coinc_inspiral', map_label, 'inspiral',
                limit=None, verbose=False)
        # convert from the output of printmissed to Result type
        for row in missed_injections:
            thisRes = Result()
            # id information
            thisRes.unique_id = idx
            thisRes.database = thisfile 
            thisRes.event_id = None
            idx += 1
            # Set the injection parameters: we'll make the injection
            # the psuedo class so we can access its attributes directly
            thisRes.set_psuedoattr_class(thisRes.injection)
            thisRes.injection.simulation_id = row.simulation_id
            thisRes.injection.approximant = row.waveform
            # ensure that m1 is always > m2
            if row.mass2 > row.mass1:
                thisRes.injection.mass1 = row.mass2
                thisRes.injection.mass2 = row.mass1
                thisRes.injection.spin1x = row.spin2x
                thisRes.injection.spin1y = row.spin2y
                thisRes.injection.spin1z = row.spin2z
                thisRes.injection.spin2x = row.spin1x
                thisRes.injection.spin2y = row.spin1y
                thisRes.injection.spin2z = row.spin1z
            else:
                thisRes.injection.mass1 = row.mass1
                thisRes.injection.mass2 = row.mass2
                thisRes.injection.spin1x = row.spin1x
                thisRes.injection.spin1y = row.spin1y
                thisRes.injection.spin1z = row.spin1z
                thisRes.injection.spin2x = row.spin2x
                thisRes.injection.spin2y = row.spin2y
                thisRes.injection.spin2z = row.spin2z
            thisRes.injection.distance = row.distance
            thisRes.injection.inclination = row.inclination
            if load_vol_weights_from_inspinj:
                # get the distribution that was used by inspinj
                r = dist 
                distr_type, distribution, r1, r2 = rdistrs[row.process_id]
                # FIXME: the scale_fac and weights are copied from 
                # randr_by_snr; these functions should be moved to library
                # location instead
                if distr_type == "chirp_dist":
                    # scale r1 and r2 by the chirp mass
                    scale_fac = (thisRes.injection.mchirp/\
                        (2.8 * 0.25**0.6))**(5./6)
                    r1 = r1*scale_fac
                    r2 = r2*scale_fac
                if distribution == "volume":
                    min_vol = (4./3)*numpy.pi*r1**3.
                    weight = (4./3)*numpy.pi*(r2**3. - r1**3.)
                elif distribution == "uniform":
                    min_vol = (4./3)*numpy.pi*r1**3.
                    weight = 4.*numpy.pi*(r2-r1) * r**2. 
                elif distribution == "log10":
                    min_vol = (4./3)*numpy.pi*r1**3.
                    weight = 4.*numpy.pi * r**3. * numpy.log(r2/r1)
                else:
                    raise ValueError("unrecognized distribution %s" %(
                        distribution))
            else:
                min_vol = row.alpha1
                weight = row.alpha2
            thisRes.injection.min_vol = min_vol
            thisRes.injection.vol_weight = vol_weight
            # since it wasn't found, the template is None
            thisRes.template = None
            # statistics: just set to bounds
            thisRes.new_snr = 0.
            thisRes.uncombined_far = thisRes.false_alarm_rate = numpy.inf
            # experiment
            thisRes.instruments_on = row.instruments_on
            # convert livetime from seconds to years
            # FIXME
            thisRes.livetime = livetime / lal.YRJUL_SI
            # get the injection distribution information
            if load_inj_distribution:
                try:
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        row.process_id]
                except KeyError:
                    # need to load the distribution
                    inj_dists[thisfile, row.process_id] = \
                        distributions.get_inspinj_distribution(connection,
                        row.process_id)
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        row.process_id]
            results.append(thisRes)
        connection.close()

    if verbose:
        print >> sys.stdout, ""
    
    # cull the results for duplicated entries and create an id_map
    results, id_map = cull_injection_results(results,
        primary_arg=cull_primary_arg,
        primary_rank_by=cull_primary_rank_by,
        secondary_arg=cull_secondary_arg,
        secondary_rank_by=cull_secondary_rank_by)
    # standdrd id map is to point to the idx, not the result
    id_map = dict([[x.simulation_id, ii] for ii,x in enumerate(results)])
    return results, id_map

def get_livetime_from_pycbc_sqlite(filenames):
    """
    Gets the total live time from a list of sqlite databases produced by the
    pycbc workflow. Live times are only added if the experiments in multiple
    filenames do not have overlapping gps end times.
    """
    sqlquery = """
        SELECT
            exp.experiment_id, exp.instruments,
            exp.gps_start_time, exp.gps_end_time,
            exsumm.veto_def_name, exsumm.datatype, exsumm.duration
        FROM
            experiment as exp
        JOIN
            experiment_summary as exsumm
        ON
            exp.experiment_id == exsumm.experiment_id
        """
    livetimes = {}
    for filename in filenames:
        connection = sqlite3.connect(filename)
        thisdict = {}
        for eid, instruments, gps_start, gps_end, vetoes, datatype, duration \
                in connection.cursor().execute(sqlquery):
            exkey = (eid, instruments, vetoes, datatype)
            this_seg = segments.segment(gps_start, gps_end)
            thisdict.setdefault(exkey, [this_seg, None])
            # for the first one, always just add the duration
            if thisdict[exkey][1] is None:
                thisdict[exkey][1] = duration 
            # if datatype is slide add the times
            elif datatype == "slide":
                thisdict[exkey][1] += duration
            # otherwise, check that the livetime is the same 
            elif duration != thisdict[exkey][1]:
                raise ValueError("unequal durations for " + exkey + \
                    "in file %s" %(filename))
        connection.close()
        # add to the master list of livetimes
        for ((_, instruments, vetoes, datatype), [this_seg, dur]) in \
                thisdict.items():
            # we'll make a dict of dict with datatype being the primary key
            livetimes.setdefault(datatype, {})
            exkey = (instruments, vetoes)
            try:
                seg_list, _ = livetimes[datatype][exkey]
            except KeyError:
                # doesn't exist, create
                seg_list = segments.segmentlist([])
                livetimes[datatype][exkey] = [seg_list, 0]
                
            # check that this_seg does not intersect with the segments
            # from the other files
            if seg_list.intersects_segment(this_seg):
                raise ValueError("Experiment (%s, %s, %s) " % exkey + \
                    "has overlapping gps times in multiple files")

            # add the duration and update the segment list
            seg_list.append(this_seg)
            seg_list.coalesce()
            livetimes[datatype][exkey][1] += dur

    return livetimes


def get_templates(filename, old_format=False):
    templates = []
    connection = sqlite3.connect(filename)
    if old_format:
        sqlquery = 'select sngl.mass1, sngl.mass2, sngl.alpha3, sngl.alpha6 from sngl_inspiral as sngl'
    else:
        sqlquery = 'select sngl.mass1, sngl.mass2, sngl.spin1z, sngl.spin2z from sngl_inspiral as sngl'
    for m1, m2, s1z, s2z in connection.cursor().execute(sqlquery):
        thisRes = Result()
        thisRes.set_psuedoattr_class(thisRes.template)
        thisRes.m1 = m1
        thisRes.m2 = m2
        thisRes.s1z = s1z
        thisRes.s2z = s2z
        templates.append(thisRes)
    connection.close()
    return templates

#
#
#   Utilities for creating html image maps
#
#
class ClickableElement:
    """
    Class to store additional information about an element in a matplotlib
    plot that is needed to create an image map area of it in an html page.

    Parameters
    ----------
    element: matplotlib element
        The element (e.g., a polygon) in a matplotlib axis overwhich we will
        create an image map area.
    shape: str
        The shape of the image map area. Valid shapes are given by the
        validshapes attribute.
    link: str
        The location of the file that the clickable area will link to.
    tag: str
        Title to appear over the clickable area during mouseover. This is
        added to the title and alt attributes of the area.
    data: Python object
        The object containing the data that the plot element was created from.
        For instance, if the plot element came from some data stored in an
        LSC Table row, data will be that row. This is useful if one needs
        to get additional information about the element, such as when
        building the html page that the clickable area points to.
    pixelcoords: numpy array
        The coordinates of the area, in display units. This should be a 2D
        array of x,y pairs. The x-values are measured from the left edge
        of the figure, the y-values from the top edge of the figure. (Note that
        this is different from matplotlib, in which the y-values are measured
        from the bottom edge.) To ensure proper formatting, pixelcoords can
        can only be set using self.set_pixelcoords().
    radius: int
        The radius of the clickable area, in display units. This should only
        be set if self's shape is circle.
    """
    element = None
    _shape = None
    _radius = None
    _pixelcoords = None
    data = None
    link = None
    tag = None
    validshapes = ['circle', 'rect', 'poly']
    
    def __init__(self, element, shape, data=None, link=None, tag=None,
        pixelcoords=None, radius=None):
        self.element = element
        self.set_shape(shape)
        if radius is not None:
            self.set_radius(radius)
        self.data = data
        self.link = link
        self.tag = tag
        if self.pixelcoords is not None:
            self.set_pixelcoords(pixelcoords)
        else:
            self._pixelcoords = None

    @property
    def shape(self):
        return self._shape

    def set_shape(self, shape):
        if not shape in self.validshapes:
            raise ValueError("unrecognized shape %s; options are %s" %(
                shape, ', '.join(self.validshapes)))
        self._shape = shape

    @property
    def radius(self):
        return self._radius

    def set_radius(self, radius):
        if self._shape != 'circle':
            raise ValueError("a radius should only be set if self's shape " +\
                "is a circle")
        self._radius = radius

    @property
    def pixelcoords(self):
        return self._pixelcoords

    def set_pixelcoords(self, coords):
        """
        Sets the pixelcoords attribute. The given coords must be a 2D array
        of x,y pairs corresponding to the pixel coordinates of the element.
        If self's shape is a rect, coords should be a 2x2 array in which the
        first row is the bottom left corner and the second row is the top
        right corner of the rectangle. If self's shape is circle, coords
        should be a 2x1 array giving the x,y location of the point. If self's
        shape is poly, coords should bw a 2xN array giving the vertices of
        the polygon, where N > 2.
        """
        # check that we have sane options
        if len(coords.shape) != 2:
            raise ValueError('coords should be a 2D array of x,y pairs')
        if self._shape == 'rect' and coords.shape != (2,2):
            raise ValueError('rect shape should have two x and two y ' +
                'coordinates, corresponding to the location of the vertices.')
        if self._shape == 'circle' and coords.shape != (1,2):
            raise ValueError('circle shape should only have a ' +
                'single x,y coordinate')
        if self._shape == 'poly' and coords.shape[0] < 3:
            raise ValueError('poly shape should have at least three x,y ' +
                'coordinates')
        # passed, set the coordinates
        self._pixelcoords = coords


class MappableFigure(matplotlib.figure.Figure):
    """
    A derived class of pyplot.figure.Figure, this adds additional
    attributes to Figure to make it easier to create an image map for it. The
    class has a get_pixelcoordinates function which will automatically get the
    pixel coordinates of all desired clickable elements formatted in the
    correct way for an image map. The class also has a savefig function that
    replace's the base class's savefig. This ensures that the proper
    coordinate transformations are carried out to get the pixel coordinates
    when the figure is renderred.

    Additional Parameters
    ---------------------
    figure_filename: str
        The name of the file the figure is saved to. This is set when savefig
        is called.
    saved_img_height: float
        The height of the figure's image in pixels when it is saved, set
        when savefig is called. This is needed for getting the proper
        coordinate locations when creating an image map. 
    saved_img_width: float
        Same as saved_img_height, but for width.
    clickable_elements: list
        A list of ClickableElement instances for which clickable areas will be
        created in an image map. All of the plot elements drawn on an axes
        that is attached to the figure.  To ensure this, a list may only be
        added via the set_clickable_elements function. To add more elements,
        use the add_clickable function.
    """
    _figure_filename = None
    _saved_img_height = None
    _saved_img_width = None
    _clickable_elements = []

    def __init__(self, *args, **kwargs):
        """
        Initializes _figure_filename, _saved_img_height, and _saved_img_width
        as None and _clickable_elements as an empty list. It then calls the
        base class's __init__ with the args and kwargs.
        """
        self._figure_filename = None
        self._saved_img_height = None
        self._saved_img_width = None
        self._clickable_elements = []
        # now initialize the figure
        super(MappableFigure, self).__init__(*args, **kwargs)

    def check_clickable_element(self, clickable):
        """
        Checks that the given clickable's element is in self's figure.
        If it is not, a ValueError is raised.
        """
        if clickable.element.figure != self:
            raise ValueError("clickable is not an element in self's figure")

    def add_clickable(self, clickable):
        self.check_clickable_element(clickable)
        self._clickable_elements.append(clickable)

    def set_clickable_elements(self, clickable_elements):
        map(self.check_clickable_element, clickable_elements)
        self._clickable_elements = clickable_elements

    @property
    def clickable_elements(self):
        return self._clickable_elements

    @property
    def saved_img_height(self):
        return self._saved_img_height

    @property
    def saved_img_width(self):
        return self._saved_img_width

    def get_pixel_coordinates(self, event):
        """
        Function to get the pixel coordinates of the clickable elements
        when the figure is saved.
        """
        self._saved_img_height = self.bbox.height
        self._saved_img_width = self.bbox.width
        for clickable in self._clickable_elements:
            ax = clickable.element.axes
            if ax not in self.axes:
                raise ValueError("clickable is not on an axes in this figure")
            pixel_coordinates = ax.transData.transform(clickable.element.xy)
            # pixel cooredinates are a 2D array; the first column is the
            # x coordinates, the second, the y coordinates
            # In html, the y-coordinate of an area is measured from the top
            # of the image, whereas matplotlib measures y from the bottom
            # of this image. For this reason, we have to adjust the y
            pixel_coordinates[:,1] *= -1.
            pixel_coordinates[:,1] += self.bbox.height
            clickable.set_pixelcoords(pixel_coordinates)

    def savefig(self, filename, **kwargs):
        """
        A wrapper around pyplot.savefig that will get the area coordinates
        of the clickable elements while saving. This needs to be done
        when the figure is rendered for saving to ensure the proper
        coordinates are retrieved. For more information, see:
        http://stackoverflow.com/a/4672015

        Also: the bbox_inches='tight' argument is not allowed if there are
        clickable elements, as this does not properly update the transformation
        matrices when the figure is trimmed (as of matplotlib version 1.4.1).
        """
        if 'bbox_inches' in kwargs and kwargs['bbox_inches'] == 'tight' and \
                self._clickable_elements != []:
            raise ValueError("bbox_inches=tight is not allowed when saving " +\
                "a mappable figure with clickable elements, as it will " +\
                "silently alter the pixel coordinates in the saved image. " +\
                "If you would like to save without the clickable elements, " +\
                "remove the clickable_elements on self [run " +\
                "self.set_clickable_elements([])], or use pyplot.savefig.")
        cid = self.canvas.mpl_connect('draw_event', self.get_pixel_coordinates)
        self._figure_filename = filename
        super(MappableFigure, self).savefig(filename, **kwargs)
        self.canvas.mpl_disconnect(cid)

    @property
    def saved_filename(self):
        """
        The filename of the saved figure. Only set when self.savefig is called.
        """
        return self._figure_filename

    def create_image_map(self, html_filename, view_width=1000,
            view_height=None):
        """
        Creates an image map of self. The pixelcoords and links of every
        element in self.clickable_elements must be populated, and
        self.saved_filename must not be None, meaning that the figure
        was saved to an image file.

        Parameters
        ----------
        html_filename: str
            The file name of the html page that the image map will be placed
            in. This is needed so that the proper relative path to the figure's
            file name can be constructed.
        view_width: {750, int}
            The width, in pixels, of the displayed image. To use the original
            size of the image, pass None.
        view_height: {None, int}
            The height, in pixels, of the displayed image. Default is None,
            in which case the height will automatically be set by the
            browser based on the view_width, such that the aspect ratio is
            preserved.

        Returns
        -------
        html: str
            HTML code describing the image link. This can be added to an
            html file to show the image map.
        """
        return _plot2imgmap(self, html_filename, view_width, view_height)
            

def figure(**kwargs):
    """
    Wrapper around pyplot's figure function that substitutes a
    matplotlib.figure.Figure with a MappableFigure in FigureClass.
    """
    if 'FigureClass' in kwargs:
        raise ValueError("this function only supports " +\
            "FigureClass=MappableFigure; use pyplot.figure() if you wish " +\
            "to use a different FigureClass")
    kwargs['FigureClass'] = MappableFigure
    return pyplot.figure(**kwargs)


def _plot2imgmap(mfig, html_filename, view_width=1000,
        view_height=None):
    """
    Creates the necessary html code to display a figure with an image map
    on an html page.

    Parameters
    ----------
    mfig: MappableFigure instance
        A instance of a MappableFigure. The clickable_elements list must
        be populated, and the figure must have been saved to a file via
        the mfig's savefig function.
    html_filename: str
        The file name of the html page that the image map will be placed
        in. This is needed so that the proper relative path to the figure's
        file name and the links can be constructed.
    view_width: {1000, int}
        The width, in pixels, of the displayed image. To use the original
        size of the image, pass None.
    view_height: {None, int}
        The height, in pixels, of the displayed image. Default is None,
        in which case the height will automatically be set by the
        browser based on the view_width, such that the aspect ratio is
        preserved.
    
    Returns
    -------
    html: str
        HTML code describing the image link. This can be added to an
        html file to show the image map.
    """
    if mfig.saved_filename is None:
        raise ValueError("mfig's saved_filename is None! Run " +
            "mfig's savefig to save the figure to disk.")
    figname = mfig.saved_filename
    # we need the fig filename relative to the html page's path
    figname = os.path.relpath(os.path.abspath(figname),
        os.path.dirname(os.path.abspath(html_filename)))
    img_height = int(mfig.saved_img_height)
    img_width = int(mfig.saved_img_width)
    # get the scale factors we need for the given view width and height
    scale_x = 1.
    scale_y = 1.
    if view_width is not None:
        scale_x = float(view_width)/img_width
        width_str = 'width="%i"' %(view_width)
        if view_height is None:
            scale_y = scale_x
    else:
        width_str = ''
    if view_height is not None:
        scale_y = float(view_height)/img_height
        if view_width is None:
            scale_x = scale_y
        height_str = 'height="%i"' %(view_height)
    else:
        height_str = ''
    scale = numpy.array([scale_x, scale_y])

    # construct the areas
    areas = []
    area_tmplt = '<area shape="%s" coords="%s" href="./%s" %s />'
    for clickable in mfig.clickable_elements:
        if clickable.shape == 'rect' or clickable.shape == 'poly':
            # for rectangle or polygon, the coordinates are just
            # x1,y1,x2,y2,...,xn,yn, where n=4 for rect
            coords = ','.join(
                (clickable.pixelcoords*scale).flatten().astype('|S32'))
        elif clickable.shape == 'circle':
            if clickable.radius is None:
                raise ValueError("shape is circle, but no radius set!")
            coords = ','.join(
                (clickable.pixelcoords*scale).flatten().astype('|S32'))
            coords += ',%i' %(clickable.radius)
        else:
            raise ValueError("unrecognized shape %s" %(clickable.shape))
        # format the link so that it is a relative path w.r.t. the html_page
        if clickable.link is None:
            raise ValueError("no link set!")
        link = os.path.relpath(os.path.abspath(clickable.link),
            os.path.dirname(os.path.abspath(html_filename)))
        # get any tags
        if clickable.tag is not None:
            tag_str = 'alt="%s" title="%s"' %(clickable.tag, clickable.tag)
        else:
            tag_str = ''
        areas.append(area_tmplt %(clickable.shape, coords, link, tag_str))

    # construct the map name from the figure's file name
    mapname = os.path.basename(figname)[:-4]

    # the map template
    tmplt = """
<div>
<img src="%s" class="mapper" usemap="#%s" border="0" %s %s />
</div>
<map name="%s">
%s
</map>
"""
    return tmplt %(figname, mapname, width_str, height_str, mapname,
        '\n'.join(areas))
