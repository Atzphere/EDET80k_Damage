import numpy as np
# params AI: 0.04641396 -0.84202717 34.0740448 (sourced from trial 1 Al)
# params TIW: [ 0.00811139  1.55514877 -7.81134315] (sourced from trial 3 TIW)

XI400_UNCERTAINTY = 0.02  # 2% works up to 100C


def quad_temp_model(x, a, b, c):
    return a * x**2 + b * x + c


def quad_temp_derivative(x, a, b, c):
    return (2 * a * x + b)


def inverse_quad_temp(x, a, b, c):
    return (-b + np.sqrt(b**2 - (4 * a * (c - x)))) / (2 * a)


class TemperatureCalibration:
    def __init__(self, converter):
        self.model_func, self.model_func_derivative, self.inverter, self.params = converter

    def true_temperature(self, T_observed, uncertainty=None):
        ''' uncertainty is absolute '''
        if uncertainty is None:
            return self.model_func(T_observed, *self.params)
        else:  # return value and propagated uncertainty
            return self.model_func(T_observed, *self.params), self.model_func_derivative(T_observed, *self.params) * uncertainty

    def original_temperature(self, T_converted):
        return self.inverter(T_converted, *self.params)


temperature_TiW = TemperatureCalibration(
    (quad_temp_model, quad_temp_derivative, inverse_quad_temp, (0.00811139, 1.55514877, -7.81134315)))

temperature_Al = TemperatureCalibration(
    (quad_temp_model, quad_temp_derivative, inverse_quad_temp, (0.04641396, -0.84202717, 34.0740448)))

maps = {"TiW": temperature_TiW, "Al": temperature_Al}
