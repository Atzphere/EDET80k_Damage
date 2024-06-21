import optris_csv as oc
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit
import numpy as np
import seaborn as sns
from temperaturemap import temperature_Al
import os

os.chdir("C:\\Users\\ssuub\\Desktop\\Damage analysis\\EDET80k_Damage\\Lasing Analysis\\apps")
print(os.getcwd())

sns.set_theme()


def fitfun(x, mu, var, a, b):
    return a * np.exp(-((x - mu)**2 / (2 * var))) + b


DSTRING = "../data/test.dat"
data = oc.OptrisDataset(DSTRING)
profile = data.writeabledata['Temperature profile 1']
darray = data.build_array_data()

start, stop = 157.4, 158.101

x_smooth = np.linspace(-1, 1, 200)
x = np.linspace(-1, 1, 20)
t = data.slice_by_time("time", start, stop)

X, T = np.meshgrid(x, t)
F = data.slice_by_time('Temperature profile 1', start, stop)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, T, F)
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

sigma_mu = []
sigma_var = []
sigma_a = []
sigma_b = []

mu_vals = []
var_vals = []
a_vals = []
b_vals = []

successful_times = []

color = iter(cm.rainbow(np.linspace(0, 1, len(t))))


for time in t:
    try:
        ydata = temperature_Al.true_temperature(np.reshape(data.slice_by_time(
            "Temperature profile 1", time, time), (20,)))
        params, ncov = curve_fit(
            fitfun, x, ydata, p0=(-0.07, 0.05, 50, 40))
        fit_uncertainties = tuple(np.sqrt(np.diag(ncov)))
        if np.max(fit_uncertainties) > 114:  # reject bad fit
            print(
                f"rejecting fit with uncertainties: {np.sqrt(np.diag(ncov))} @ {time}")
        else:  # plot good fits
            c = next(color)  # r a i n b o w
            print(
                f"keeping fit with uncertainties: {np.sqrt(np.diag(ncov))} @ {time}")
            ax.plot(x_smooth, fitfun(x_smooth, *params), c=c)
            ax.scatter(x, ydata, label=time, c=c)
            ax2.errorbar(x, ydata - fitfun(x, *params), yerr=0.5, label=time, c=c)
            mu_delta, var_delta, a_delta, b_delta = fit_uncertainties
            mu, var, a, b = params
            sigma_mu.append(mu_delta)  # record data for data
            sigma_var.append(var_delta)
            sigma_a.append(a_delta)
            sigma_b.append(b_delta)
            mu_vals.append(mu)
            var_vals.append(var)
            a_vals.append(a)
            b_vals.append(b)
            successful_times.append(time)

    except RuntimeError:
        print(f"Fit failed at time {time}")

print(np.mean(mu_vals))

# ax.plot(x, fitfun(x, -0.07, 0.05, 50, 40))
ax.legend()
ax.set_xlabel("x (arbitrary)")
ax.set_ylabel("T (C)")
ax.set_title("Temperature profile for 3.0A 1S laser pulse")

ax2.legend(loc=1)
ax2.set_xlabel("x (arbitrary)")
ax2.set_ylabel("Temperature")
ax2.set_title("Residuals")

fig.show()
fig2.show()

fig, ax = plt.subplots()

ax.set_xlabel("Time (s)")
ax.set_title("Parameters w.r.t. time: extrinsic")
ax.errorbar(successful_times, a_vals, yerr=sigma_a, label="a")
ax.errorbar(successful_times, b_vals, yerr=sigma_b, label="b")
ax.legend()

plt.show()

fig, ax = plt.subplots()

ax.set_xlabel("Time (s)")
ax.set_title("Parameters w.r.t. time: intrinsic")
ax.errorbar(successful_times, mu_vals, yerr=sigma_mu, label="mu")
ax.errorbar(successful_times, var_vals, yerr=sigma_var, label="var")
ax.legend()

plt.show()

print('done')
