import numpy as np
from numpy.polynomial import Polynomial
import logging
# params AI: 0.04641396 -0.84202717 34.0740448 (sourced from trial 1 Al)
# params TIW: [ 0.00811139  1.55514877 -7.81134315] (sourced from trial 3 TIW)

# new more aggresive params:
# TiW: 109.97375249530333 + 71.23636313978952 x**1 + 16.011679663543163 x**2 + 17.63904545775844 x**3 - 3.6155737011062103 x**4
# Al: 95.77908922235851 + 64.04110673158722 x**1 + 19.923303870421066 x**2 + 14.281999148143642 x**3 - 0.3250204326056052 x**4


XI400_UNCERTAINTY = 0.02  # 2% works past 100C
XI400_LOWTEMP_UNCERTAINTY = 2  # under 100C


def quad_temp_model(x, a, b, c):
    return a * x**2 + b * x + c


def quad_temp_derivative(x, a, b, c):
    return (2 * a * x + b)


def inverse_quad_temp(x, a, b, c):
    return (-b + np.sqrt(b**2 - (4 * a * (c - x)))) / (2 * a)


class TemperatureCalibration:
    def __init__(self, converter, domain=None):
        self.model_func, self.model_func_derivative, self.params, = converter
        self.domain = domain

    def true_temperature(self, T_observed, uncertainty=None, auto_uncertainty=False):
        ''' uncertainty is absolute '''
        lowerbound, upperbound = self.domain
        unstable_value = (T_observed < lowerbound) | (T_observed > upperbound)
        if self.domain is not None and np.any(unstable_value):
            logging.warning(f"Temperature(s) {T_observed[unstable_value]} is beyond nominal temperature mapping range.")
        T_converted = self.model_func(T_observed, *self.params)
        if uncertainty is None and not auto_uncertainty:
            return T_converted
        elif auto_uncertainty:
            return T_converted, self.model_func_derivative(T_observed, *self.params) * get_base_uncertainties(T_observed)
        else:  # return value and propagated uncertainty
            return T_converted, self.model_func_derivative(T_observed, *self.params) * uncertainty


class BlackBodyTC():
    def true_temperature(self, T, *args):
        return T


class TemperatureCalibrationInvertible(TemperatureCalibration):
    def __init__(self, converter, domain, inverter):
        super().__init__(converter, domain)
        self.inverter = inverter

    def original_temperature(self, T_converted):
        return self.inverter(T_converted, *self.params)


class TemperatureCalibrationPolyomial(TemperatureCalibration):
    def __init__(self, poly, domain):
        self.model_func = poly
        self.domain = domain
        self.params = ()
        self.model_func_derivative = poly.deriv()


def get_base_uncertainties(data):
    uncertainties = XI400_LOWTEMP_UNCERTAINTY * np.ones(np.shape(data))
    hightemp_uncertainties = (data * XI400_UNCERTAINTY)
    uncertainties[hightemp_uncertainties >
                  uncertainties] = hightemp_uncertainties[hightemp_uncertainties > uncertainties]
    return uncertainties


temperature_TiW_old = TemperatureCalibrationInvertible(
    (quad_temp_model, quad_temp_derivative, (0.00811139, 1.55514877, -7.81134315)), (0, 300), inverse_quad_temp)

temperature_Al_old = TemperatureCalibrationInvertible(
    (quad_temp_model, quad_temp_derivative, (0.04641396, -0.84202717, 34.0740448)), (0, 300), inverse_quad_temp)

TiW_model = Polynomial([109.97375249530333, 71.23636313978952, 16.011679663543163, 17.63904545775844, -3.6155737011062103], domain=(24.05, 76.07))
Al_model = Polynomial([95.77908922235851, 64.04110673158722, 19.923303870421066, 14.281999148143642, -0.3250204326056052], domain=(24.04, 59.04))


blackbody = BlackBodyTC()

temperature_TiW = TemperatureCalibrationPolyomial(TiW_model, domain=(24.05, 76.07))
temperature_Al = TemperatureCalibrationPolyomial(Al_model, domain=(24.04, 59.04))


maps = {"TiW_alt": temperature_TiW_old, "Al_alt": temperature_Al_old,
        "TiW": temperature_TiW, "Al": temperature_Al,
        "Blackbody": blackbody}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(24, 100), maps["Al"].true_temperature(np.linspace(24, 100)))
    plt.show()