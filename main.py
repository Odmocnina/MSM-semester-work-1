import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import sympy as sp


def get_constant_c(func_f: sp.Expr, x_interval: tuple, y_interval: tuple, x: sp.Symbol, y: sp.Symbol, c: sp.Symbol,
                   expected_value: float = 1) -> float | None:
    """
    Function for getting the c value, taken from that the formla: ∫∫ c * f(x,y) dx dy = 1, converted to c = 1 / ∫∫ f(x,y) dx dy

    Parameters
    ----------
    func_f: sp.Expr
        the function f(x,y) to be integrated (containing the unknown constant c)
    x_interval: tuple
        the interval for x, given as a tuple (x_min, x_max)
    y_interval: tuple
        the interval for y, given as a tuple (y_min, y_max)
    x: sp.Symbol
        the symbol for the variable x
    y: sp.Symbol
        the symbol for the variable y
    c: sp.Symbol
        the symbol of the unknown constant c that is solved for
    expected_value: float
        the expected value of the integral (default is 1)

    Returns
    -------
    sp.Expr | None
        the value of c that satisfies the equation, or None if there is no solution
    """
    # calculate the integral and then return the c value
    double_integral = sp.integrate(func_f,
                                   (y, y_interval[0], y_interval[1]),
                                   (x, x_interval[0], x_interval[1]))       # result of this is function with 1 parameter c, so we can solve for c
    c_equation = sp.Eq(double_integral, expected_value)  # create eqasion (result of the dobule integral = expected_value (usualy 1))
    c_solution: list[float] = sp.solve(c_equation, c)  # solve for c, sp.solve returns an empty list if there is no solution
    return c_solution[0] if c_solution else None

def get_marginal_density(func: sp.Expr, interval: tuple, variable: sp.Symbol) -> sp.Expr:
    """
    Calculates the marginal density function from the joint density function. The formula is f_X(x) = ∫ f(x,y) dy (or f_Y(y) = ∫ f(x,y) dx),
    so we integrate out the other variable over the whole range where the joint density is nonzero.
    The interval can depend on the remaining variable (e.g. y in (x, 2) for f_X, or x in (1, y) for f_Y).

    Parameters
    ----------
    func: sp.Expr
        the joint density function f(x,y) (with the constant c already substituted)
    interval: tuple
        the interval of the variable that is integrated out, given as a tuple (min, max);
        the bounds may contain the other variable, e.g. (x, 2) or (1, y)
    variable: sp.Symbol
        the variable to integrate over (the one that is eliminated, e.g., y for f_X(x) or x for f_Y(y))

    Returns
    -------
    sp.Expr
        the marginal density function f_X(x) or f_Y(y)
    """
    marginal_distribution = sp.integrate(func, (variable, interval[0], interval[1]))
    return marginal_distribution

def get_distribution_function(density_func: sp.Expr, interval: tuple, variable: sp.Symbol) -> sp.Expr:
    """
    Calculates the cumulative distribution function (CDF) from a density function. The formula is the ∫ f(x) dx where the f(x) is the marginal density function.
    The interval on witch is integrated is the lowest value that it can be and the top is not used.

    Parameters
    ----------
    density_func: sp.Expr
        the marginal density function f(x) or f(y)
    interval: tuple
        the interval for the variable, we need the lower bound (interval[0])
    variable: sp.Symbol
        the variable to integrate up to (e.g., x or y)

    Returns
    -------
    sp.Expr
        the distribution function F(x) or F(y)
    """
    distribution_function = sp.integrate(density_func, (variable, interval[0], variable))
    return distribution_function

def generate_random_samples(n_samples: int, fnc: sp.Expr, x: sp.Symbol, y: sp.Symbol,
                            x_bounds: tuple, y_bounds: tuple, max_point: dict,
                            domain_condition) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random samples from a given 2D distribution using rejection sampling. Basily 2 numbers are generated, it is checked if they are alright with out intervals
    (fr us that y > x) and also with them are generated random z values, which are then compared to the results of the function and if the random z are smaller the numbers
    are accepted. These z values are generated so that we can simulate the distribution becouse if it was not there it would be even distribution.

    Parameters
    ----------
    n_samples: int
        number of random points to be generated
    fnc: sp.Expr
        distribution fuction we are inverstigating and where we want to generate the points
    x: sp.Symbol
        the symbol for the variable x
    y: sp.Symbol
        the symbol for the variable y
    x_bounds: tuple
        the interval (min, max) for generating x candidates
    y_bounds: tuple
        the interval (min, max) for generating y candidates
    max_point: dict
        a dictionary containing the coordinates of the function's maximum value (e.g., {x: 2, y: 2}), later used for restrition of the generated values so that the
        distribution is simulated
    domain_condition: callable
        a function or lambda taking x and y arrays, returning a boolean mask of points that satisfy the domain rules, this is for if the itervals of the 2 varibles are
        dependant on each other for exapmle x is on interval 1-2 and y is on interval x-2 (y is bigger then x)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        two numpy arrays containing the generated x and y samples
    """

    # paramters for generation
    x_samples = []
    y_samples = []

    # this creates a fuction (program function) from mathematical function
    f_num = sp.lambdify((x, y), fnc, "numpy")

    # get the max value of the fuction using the provided max_point parameter, its in dictionary so that the
    # so that it can be added in parameters and its not dependent on what is the function
    max_f = float(fnc.subs(max_point))

    print("Generation of random samples...")
    while len(x_samples) < n_samples:
        # we generate the x and y values for the function within the given boundaries
        x_cand = np.random.uniform(x_bounds[0], x_bounds[1], n_samples)
        y_cand = np.random.uniform(y_bounds[0], y_bounds[1], n_samples)

        # we generate a z value that will filter out the small values gotten from the function
        u_cand = np.random.uniform(0, max_f, n_samples)

        # cut the values that do not ccomplish the conditin (defined by domain_condition)
        # this is if for example the interval of 1 variable is limited by second one (x is 1 - 2 and y is x - 2 (y is bigger then x))
        domain_mask = domain_condition(x_cand, y_cand)

        # get the values from the function
        f_vals = f_num(x_cand[domain_mask], y_cand[domain_mask])

        # take the filtered out values from the first condition and look if the randomly gen. value is smaller then the value gotten fro mfuction
        # and if it is the accept it, this is done so that the point that are being generated simulate the distibution of our function
        # if this would not be here it would basicly be a even distribution becouse everything would be of the same propability
        accept_mask = u_cand[domain_mask] <= f_vals

        # add the samples that passed
        x_samples.extend(x_cand[domain_mask][accept_mask])
        y_samples.extend(y_cand[domain_mask][accept_mask])

    # cut the array to our wanted amount of samples
    x_samples = np.array(x_samples[:n_samples])
    y_samples = np.array(y_samples[:n_samples])
    print(f"Succesfully generated {n_samples} random samples.")

    return x_samples, y_samples

def estimate_characteristics(x_samples: np.ndarray, y_samples: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Estimates the means, variances and covariance from the generated samples and constructs the confidence intervals
    for the means. The interval is avrage +- t_crit * sqrt(var / n), where t_crit is the quantile of the Student distribution.

    Parameters
    ----------
    x_samples: np.ndarray
        generated samples of the variable X
    y_samples: np.ndarray
        generated samples of the variable Y
    alpha: float
        the uncertainty of the confidence interval (default is 0.05, so we are 95% sure that the real mean is in the interval)

    Returns
    -------
    dict
        dictionary with keys n, mean_x, mean_y, var_x, var_y, cov_xy, cov_matrix, ci_x, ci_y
        (ci_x and ci_y are tuples (lower bound, upper bound))
    """
    n = len(x_samples)

    # the avrages and variations
    mean_x = np.mean(x_samples)
    mean_y = np.mean(y_samples)
    # ddof=1 is a degree of freedom, that if this would not be here the fomula would use N in division, now N - 1 is used
    # this is for getting the unbiased variation becouse randomly generated data are always going to have bias and we got average form
    # randomly generated data
    var_x = np.var(x_samples, ddof=1)
    var_y = np.var(y_samples, ddof=1)

    # create a 2x2 variation matrix
    cov_matrix = np.cov(x_samples, y_samples)

    # this is becouse we were doing random choosing we cant be sure that the point werent genrated
    # randomly only in one part of the set (for exaple we would want an avrage male height se we be messuring 1000 people but we could randomly
    # mesure 999 basketball players and Jonas which would make the avrage and all another stats pretty unaccurate)
    # so we need to add a amount of uncertainty (alpha) so when the aplha is 0,05 we are 95% (0,95) sure that their avrage is close to the overall avrage
    # The aplha is a paramter that says that how ciartan we want to be that in the interval is the avrage of the whole set
    # so that when for the example with the height we get avrage of 1000 people, and we want to have a interval in witch is a avrage of the
    # height of all poeple in the world
    # basicly this is done so that we get from one value got from the random picking we get interval

    # get the quantiol of the student distribution (searches table), it gives it 1 - alpha/2 and n - 1 samples becouse it is in the formula
    t_crit = st.t.ppf(1 - alpha/2, df=n-1)

    # this impments the formula for the interval bounaries of boundary = avrage +- t_form_student_dis * (sqr(var / number_of_samples))
    half_width_x = t_crit * np.sqrt(var_x / n)
    half_width_y = t_crit * np.sqrt(var_y / n)

    return {
        "n": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "var_x": var_x,
        "var_y": var_y,
        "cov_xy": cov_matrix[0, 1],
        "cov_matrix": cov_matrix,
        "ci_x": (mean_x - half_width_x, mean_x + half_width_x),
        "ci_y": (mean_y - half_width_y, mean_y + half_width_y),
    }

'''def generate_random_samples_old(n_samples: int, fnc: sp.Expr, x: sp.Symbol, y: sp.Symbol) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random samples from a given 2D distribution using rejection sampling. Basily 2 numbers are generated, it is checked if they are alright with out intervals
    (fr us that y > x) and also with them are generated random z values, which are then compared to the results of the function and if the random z are smaller the numbers
    are accepted. These z values are generated so that we can simulate the distribution becouse if it was not there it would be even distribution.

    Parameters
    ----------
    n_samples: int
        number of random points to be generated
    fnc: sp.Expr
        distribution fuction we are inverstigating and where we want to generate the points

    """
    # paramters for generation

    x_samples = []
    y_samples = []

        # this creates a fuction (program function) from mathematical function
    f_num = sp.lambdify((x, y), fnc, "numpy")

    # get the max value of the fuction (the x: 2, y: 2 is becouse the fnc is always getting bigger the bigger number it gets)
    max_f = float(fnc.subs({x: 2, y: 2}))

    print("Generation of random samples...")
    while len(x_samples) < n_samples:
        # we generate the x and y values for the function
        x_cand = np.random.uniform(1, 2, n_samples)
        y_cand = np.random.uniform(1, 2, n_samples)

        # we generate a z value that will filter out the small values gotten from the function
        u_cand = np.random.uniform(0, max_f, n_samples)

        # cut the values that do not ccomplish the conditin where y >= x (intervals)
        domain_mask = y_cand >= x_cand

        # get the values from the function
        f_vals = f_num(x_cand[domain_mask], y_cand[domain_mask])

        # take the filtered out values from the first condition and look if the randomly gen. value is smaller then the value gotten fro mfuction
        # and if it is the accept it, this is done so that the point that are being generated simulate the distibution of our function
        # if this would not be here it would basicly be a even distribution becouse everything would be of the same propability
        accept_mask = u_cand[domain_mask] <= f_vals

        # add the samples that passed
        x_samples.extend(x_cand[domain_mask][accept_mask])
        y_samples.extend(y_cand[domain_mask][accept_mask])

    # cut the array to our wanted amount of samples
    x_samples = np.array(x_samples[:n_samples])
    y_samples = np.array(y_samples[:n_samples])
    print(f"Succesfully generated {n_samples} random samples.")

    return x_samples, y_samples'''

class Tee:
    """
    Writes everything to the console and to a file at the same time (like the unix command tee).
    It is used as a replacement of sys.stdout, so every print() goes both to the console and to the log file.

    Parameters
    ----------
    console: TextIO
        the original output stream (sys.stdout)
    file: TextIO
        the opened log file
    """
    def __init__(self, console: TextIO, file: TextIO):
        self.console = console
        self.file = file

    def write(self, text: str) -> None:
        self.console.write(text)
        self.file.write(text)

    def flush(self) -> None:
        self.console.flush()
        self.file.flush()

if __name__ == "__main__":
    # all figures are also saved as pdf into this folder (next to this script), it is created if it does not exist
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # everything that is printed is also written into the log file (it is overwritten on every run),
    # utf-8 is needed because of the characters like ≈ or ∫ in the prints
    log_file = open(output_dir / "log.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = cast(TextIO, Tee(original_stdout, log_file))      # cast is only for the type checker, Tee behaves like TextIO
    print(f"Run from {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ================================================================================================================
    # ================================================= 1. PART ======================================================
    print("\n" + "="*50)
    print("1. PART: Constant c value")
    print("="*50)

    x, y, c = sp.symbols('x y c')
    fnc: sp.Expr = c * x * y                   # function to be investigated

    x_interval = (1, 2)              # intervals for random variables
    y_interval = (x, 2)

    c_value = get_constant_c(fnc, x_interval, y_interval, x, y, c)
    print("Calculating the constant c for the function f(x,y) = c * x * y over the intervals x in [1, 2] and y in [x, 2]...")
    print(f"The constant c is: {c_value}")

    fnc = cast(sp.Expr, fnc.subs(c, c_value))  # update the function with newlz get c value substitute the value of c into the function

    # ================================================== VIZULAIZATION OF THE FUNCTION ================================

    f = sp.lambdify((x, y), fnc, "numpy")

    xs = np.linspace(1, 2, 100)
    ys = np.linspace(1, 2, 100)
    X, Y = np.meshgrid(xs, ys)

    Z = f(X, Y)
    Z = np.where(Y >= X, Z, np.nan)

    fig = plt.figure("Část 1: Sdružená hustota", figsize=(10, 8))      # the first argument is the window title
    fig.suptitle(f"Část 1: Sdružená hustota f(x, y) = {fnc}", fontsize=14)
    ax1 = fig.add_subplot(111, projection="3d")

    ax1.plot_surface(X, Y, Z, cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")

    ax1.set_box_aspect(None, zoom=1.2)                      # zooms on 3D graph
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

    fig.savefig(output_dir / "part1_joint_density.pdf")
    plt.show()

    # ================================================================================================================
    # ================================================= 2. PART ======================================================
    print("\n" + "="*50)
    print("2. PART: Marginal density and distribution")
    print("="*50)

    # for x function we intergate y on what y can be (normal iterval)
    marginal_x = get_marginal_density(fnc, y_interval, y)
    marginal_y = get_marginal_density(fnc, (1, y), x)     # but here we must take into considiration that y is dependent on x (x < y) so we must put the itervat from
    # to y so that x > y doesnt happen
    print(f"The marginal density function for x is: {marginal_x}")
    print(f"The marginal density function for y is: {marginal_y}")

    # but here the itervals for distribution are all values that the variable can have so 1 to 2
    distribution_x = get_distribution_function(marginal_x, (1, 2), x)
    distribution_y = get_distribution_function(marginal_y, (1, 2), y)
    print(f"The marginal distribution function for x is: {distribution_x}")
    print(f"The marginal distribution function for y is: {distribution_y}")

    # ================================================== VIZULAIZATION OF MARGINAL FUNCTIONS ==========================

    f_x = sp.lambdify(x, marginal_x, "numpy")
    f_y = sp.lambdify(y, marginal_y, "numpy")
    F_x = sp.lambdify(x, distribution_x, "numpy")
    F_y = sp.lambdify(y, distribution_y, "numpy")

    t = np.linspace(1, 2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), num="Část 2: Marginální funkce")     # num is the window title
    fig.suptitle("Část 2: Marginální hustoty a distribuční funkce", fontsize=14)

    # Density
    axes[0].plot(t, f_x(t), label="f_X(x)")
    axes[0].plot(t, f_y(t), label="f_Y(y)")
    axes[0].set_title("Marginální hustota")
    axes[0].legend()
    axes[0].grid(True)

    # distribution
    axes[1].plot(t, F_x(t), label="F_X(x)")
    axes[1].plot(t, F_y(t), label="F_Y(y)")
    axes[1].set_title("Marginální distribuční funkce")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fig.savefig(output_dir / "part2_marginal_functions.pdf")
    plt.show()

    # ================================================================================================================
    # ================================================= 3. PART ======================================================
    print("\n" + "="*50)
    print("3. PART: Function characterisics")
    print("="*50)


    # middle vlaues: E[X] = ∫ x·f_X(x) dx, E[Y] = ∫ y·f_Y(y) dy, the iterval is all values that the variable can have
    E_x = sp.integrate(x * marginal_x, (x, 1, 2))
    E_y = sp.integrate(y * marginal_y, (y, 1, 2))

    # second moments, used for variation: E[X²], E[Y²], the iterval is all values that the variable can have
    E_x2 = sp.integrate(x**2 * marginal_x, (x, 1, 2))
    E_y2 = sp.integrate(y**2 * marginal_y, (y, 1, 2))

    # variation: Var[X] = E[X²] - (E[X])²
    var_x = E_x2 - E_x**2
    var_y = E_y2 - E_y**2

    # toggether moment, used for covarience E[XY] = ∫∫ x·y·f(x,y) dy dx  (stejné meze jako při výpočtu c)
    E_xy = sp.integrate(x * y * fnc, (y, y_interval[0], y_interval[1]), (x, x_interval[0], x_interval[1]))

    # covariance: cov(X,Y) = E[XY] - E[X]·E[Y]
    cov_xy = E_xy - E_x * E_y

    # covariation matrix
    cov_matrix = sp.Matrix([[var_x, cov_xy],
                            [cov_xy, var_y]])

    # ================================================ Print results =================================================

    print(f"E[X] = {E_x} ≈ {float(E_x):.4f}")
    print(f"E[Y] = {E_y} ≈ {float(E_y):.4f}")
    print(f"Var[X] = {var_x} ≈ {float(var_x):.5f}")
    print(f"Var[Y] = {var_y} ≈ {float(var_y):.5f}")
    print(f"cov(X,Y) = {cov_xy} ≈ {float(cov_xy):.5f}")
    print("Variation matrix:")
    sp.pprint(cov_matrix)

    # ================================================================================================================
    # ================================================= 4. PART ======================================================
    print("\n" + "="*50)
    print("4. PART: Data Generation")
    print("="*50)
    n_samples = 1000

    interval_condition = lambda x_val, y_val: y_val >= x_val

    # x_samples, y_samples = generate_random_samples(n_samples, fnc, x, y)
    x_samples, y_samples = generate_random_samples(
        n_samples=n_samples,
        fnc=fnc,
        x=x,
        y=y,
        x_bounds=x_interval,
        y_bounds=x_interval,
        max_point={x: 2, y: 2},
        domain_condition=interval_condition
    )

    # ================================================================================================================
    # ================================================= 5. PART ======================================================
    print("\n" + "="*50)
    print("5. PART: Findings from generated data")
    print("="*50)

    alpha = 0.05            # uncertainty of the confidence intervals (95% confidence)

    est = estimate_characteristics(x_samples, y_samples, alpha)
    ci_x = est["ci_x"]
    ci_y = est["ci_y"]

    print("Point aproximation from the function:")
    print(f"E[X] ≈ {est['mean_x']:.4f}")
    print(f"E[Y] ≈ {est['mean_y']:.4f}")
    print(f"Var[X] ≈ {est['var_x']:.5f}")
    print(f"Var[Y] ≈ {est['var_y']:.5f}")
    print(f"cov(X,Y) ≈ {est['cov_xy']:.5f}")
    print("Variation matrix done from aproximation")
    print(est["cov_matrix"])

    conf_pct = (1 - alpha) * 100

    print(f"\nInterval guess for avrage ({conf_pct:.0f}% confident interval):")
    print(f"IS for E[X]: ({ci_x[0]:.4f}, {ci_x[1]:.4f})")
    print(f"IS for E[Y]: ({ci_y[0]:.4f}, {ci_y[1]:.4f})")

    # ================================================================================================================
    # ================================================= 6. PART ======================================================
    print("\n" + "="*50)
    print("6. PART: Comparison with theoretical functions")
    print("="*50)

    # new figure is needed, the one from part 2 was already closed by plt.show()
    # top row: histograms vs. marginal densities, bottom row: empirical CDFs vs. marginal distribution functions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), num="Část 6: Porovnání s teorií")     # num is the window title
    fig.suptitle(f"Část 6: Vygenerovaná data (n = {n_samples}) vs. teoretické funkce", fontsize=14)

    # histogram vs. theoretical marginal density, density=True normalizes the histogram so that its area is 1,
    # otherwise it would show counts and could not be compared with the density function
    axes[0, 0].hist(x_samples, bins=30, density=True, alpha=0.6, label="histogram X")
    axes[0, 0].plot(t, f_x(t), "r", label="f_X(x)")
    axes[0, 0].set_title("X: histogram vs. marginální hustota")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].hist(y_samples, bins=30, density=True, alpha=0.6, label="histogram Y")
    axes[0, 1].plot(t, f_y(t), "r", label="f_Y(y)")
    axes[0, 1].set_title("Y: histogram vs. marginální hustota")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # empirical distribution function (ECDF) vs. theoretical marginal distribution function,
    # ECDF in point t is the fraction of samples that are <= t, the formula is F_n(t) = (1/n) * Σ 1(X_i <= t),
    # where n is the number of samples and 1(X_i <= t) is 1 if the sample X_i is <= t, otherwise 0
    axes[1, 0].ecdf(x_samples, label="ECDF X")
    axes[1, 0].plot(t, F_x(t), "r", label="F_X(x)")
    axes[1, 0].set_title("X: empirická vs. marginální distribuční funkce")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].ecdf(y_samples, label="ECDF Y")
    axes[1, 1].plot(t, F_y(t), "r", label="F_Y(y)")
    axes[1, 1].set_title("Y: empirická vs. marginální distribuční funkce")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    fig.savefig(output_dir / "part6_comparison_with_theory.pdf")
    plt.show()

    # ================================================================================================================
    # ================================================= 7. PART ======================================================
    print("\n" + "="*50)
    print("7. PART: Dependence of the estimates on the sample size")
    print("="*50)

    # sample sizes for which the data are generated again and the estimates are computed
    n_values = [10, 100, 1000, 10000, 100000]

    results = []
    for n in n_values:
        xs_n, ys_n = generate_random_samples(
            n_samples=n,
            fnc=fnc,
            x=x,
            y=y,
            x_bounds=x_interval,
            y_bounds=x_interval,
            max_point={x: 2, y: 2},
            domain_condition=interval_condition
        )
        results.append(estimate_characteristics(xs_n, ys_n, alpha))

    # theoretical values from part 3 converted from sympy to float, so they can be compared with the estimates
    E_x_f, E_y_f = float(E_x), float(E_y)
    var_x_f, var_y_f, cov_xy_f = float(var_x), float(var_y), float(cov_xy)

    # ================================================ Print results =================================================

    print(f"\nEstimates of the means and their {conf_pct:.0f}% confidence intervals:")
    print(f"{'n':>8} | {'E[X] est':>9} | {'IS for E[X]':^16} | {'width':>8} | {'error':>8} || "
          f"{'E[Y] est':>9} | {'IS for E[Y]':^16} | {'width':>8} | {'error':>8}")
    for r in results:
        width_x = r["ci_x"][1] - r["ci_x"][0]
        width_y = r["ci_y"][1] - r["ci_y"][0]
        print(f"{r['n']:>8} | {r['mean_x']:>9.4f} | ({r['ci_x'][0]:.4f}, {r['ci_x'][1]:.4f}) | {width_x:>8.5f} | "
              f"{abs(r['mean_x'] - E_x_f):>8.5f} || "
              f"{r['mean_y']:>9.4f} | ({r['ci_y'][0]:.4f}, {r['ci_y'][1]:.4f}) | {width_y:>8.5f} | "
              f"{abs(r['mean_y'] - E_y_f):>8.5f}")
    print(f"Theoretical values: E[X] = {E_x_f:.4f}, E[Y] = {E_y_f:.4f}")

    print("\nEstimates of the variances and covariance:")
    print(f"{'n':>8} | {'Var[X] est':>10} | {'error':>8} | {'Var[Y] est':>10} | {'error':>8} | {'cov est':>9} | {'error':>8}")
    for r in results:
        print(f"{r['n']:>8} | {r['var_x']:>10.5f} | {abs(r['var_x'] - var_x_f):>8.5f} | "
              f"{r['var_y']:>10.5f} | {abs(r['var_y'] - var_y_f):>8.5f} | "
              f"{r['cov_xy']:>9.5f} | {abs(r['cov_xy'] - cov_xy_f):>8.5f}")
    print(f"Theoretical values: Var[X] = {var_x_f:.5f}, Var[Y] = {var_y_f:.5f}, cov(X,Y) = {cov_xy_f:.5f}")

    # ================================================== VIZULAIZATION OF THE DEPENDENCE ON n ========================

    ns = np.array([r["n"] for r in results])
    means_x = np.array([r["mean_x"] for r in results])
    means_y = np.array([r["mean_y"] for r in results])
    # half widths of the intervals, used as error bars (the interval is mean +- half width)
    half_x = np.array([(r["ci_x"][1] - r["ci_x"][0]) / 2 for r in results])
    half_y = np.array([(r["ci_y"][1] - r["ci_y"][0]) / 2 for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), num="Část 7: Závislost na n")     # num is the window title
    fig.suptitle(f"Část 7: Závislost odhadů na rozsahu výběru n ({conf_pct:.0f}% intervaly spolehlivosti)", fontsize=14)

    # estimates of the means with their confidence intervals, the dashed line is the theoretical value,
    # the x axis is logarithmic because the n values grow 10 times each step
    axes[0].errorbar(ns, means_x, yerr=half_x, fmt="o", capsize=4, label="odhad E[X]")
    axes[0].axhline(E_x_f, color="r", linestyle="--", label="teoretická E[X]")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n")
    axes[0].set_title("Odhad E[X]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].errorbar(ns, means_y, yerr=half_y, fmt="o", capsize=4, label="odhad E[Y]")
    axes[1].axhline(E_y_f, color="r", linestyle="--", label="teoretická E[Y]")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("n")
    axes[1].set_title("Odhad E[Y]")
    axes[1].legend()
    axes[1].grid(True)

    # width of the interval is 2 * t_crit * sqrt(var / n), so it decreases like 1 / sqrt(n),
    # in log-log scale this is a line with the slope -1/2 (10 times more data = about 3.16 times narrower interval)
    axes[2].loglog(ns, 2 * half_x, "o-", label="šířka IS pro E[X]")
    axes[2].loglog(ns, 2 * half_y, "s-", label="šířka IS pro E[Y]")
    axes[2].loglog(ns, 2 * half_x[0] * np.sqrt(ns[0] / ns), "k--", label="~ 1/√n")
    axes[2].set_xlabel("n")
    axes[2].set_title("Šířka intervalu spolehlivosti")
    axes[2].legend()
    axes[2].grid(True, which="both")

    plt.tight_layout()
    fig.savefig(output_dir / "part7_dependence_on_n.pdf")
    plt.show()

    # return the original console output and close the log file
    sys.stdout = original_stdout
    log_file.close()
