# params AI: 0.04641396 -0.84202717 34.0740448 (sourced from trial 1 Al)
# params TIW: [ 0.00811139  1.55514877 -7.81134315] (sourced from trial 3 TIW)


def quad_temp_model(x, a, b, c):
    return a * x**2 + b * x + c


class TemperatureCalibration:
    def __init__(self, converter):
        self.model_func, self.params = converter

    def true_temperature(self, T_observed):
        return self.model_func(T_observed, *self.params)


temperature_TiW = TemperatureCalibration(
    (quad_temp_model, (0.00811139, 1.55514877, -7.81134315)))

temperature_Al = TemperatureCalibration(
    (quad_temp_model, (0.04641396, -0.84202717, 34.0740448)))

maps = {"TiW" : temperature_TiW, "Al" : temperature_Al}
