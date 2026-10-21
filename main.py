from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy.stats as st

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

if __name__ == "__main__":
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

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection="3d")

    ax1.plot_surface(X, Y, Z, cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")
    ax1.set_title("Density function f(x, y)")

    ax1.set_box_aspect(None, zoom=1.2)                      # zooms on 3D graph
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Density
    axes[0].plot(t, f_x(t), label="f_X(x)")
    axes[0].plot(t, f_y(t), label="f_Y(y)")
    axes[0].set_title("Marginal density")
    axes[0].legend()
    axes[0].grid(True)

    # distribution
    axes[1].plot(t, F_x(t), label="F_X(x)")
    axes[1].plot(t, F_y(t), label="F_Y(y)")
    axes[1].set_title("Marginal distribution")
    axes[1].legend()
    axes[1].grid(True)

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

    # the avrages and variations
    mean_x_est = np.mean(x_samples)
    mean_y_est = np.mean(y_samples)
    # ddof=1 is a degree of freedom, ift his woulnd not be here the fomula would use N in division, now N - 1 is used
    # this is for getting the unbiased variation becouse randomly generated data are always going to have bias amd we got avrage form
    # randomly generated data
    var_x_est = np.var(x_samples, ddof=1) 
    var_y_est = np.var(y_samples, ddof=1)
    
    # create a 2x2 vriation matrix
    cov_matrix_est = np.cov(x_samples, y_samples)
    cov_xy_est = cov_matrix_est[0, 1]

    print("Point aproximation from the function:")
    print(f"E[X] ≈ {mean_x_est:.4f}")
    print(f"E[Y] ≈ {mean_y_est:.4f}")
    print(f"Var[X] ≈ {var_x_est:.5f}")
    print(f"Var[Y] ≈ {var_y_est:.5f}")
    print(f"cov(X,Y) ≈ {cov_xy_est:.5f}")
    print("Variation matrix done from aproximation")
    print(cov_matrix_est)

    # this is becouse we were doing random choosing we cant be sure that the point werent genrated
    # randomly only in one part of the set (for exaple we would want an avrage male height se we be messuring 1000 people but we could randomly 
    # mesure 999 basketball players and Jonas which would make the avrage and all another stats pretty unaccurate)
    # so we need to add a amount of uncertainty (alpha) so when the aplha is 0,05 we are 95% (0,95) sure that their avrage is close to the overall avrage
    # The aplha is a paramter that says that how ciartan we want to be that in the interval is the avrage of the whole set
    # so that when for the example with the height we get avrage of 1000 people, and we want to have a interval in witch is a avrage of the
    # height of all poeple in the world
    # basicly this is done so that we get from one value got from the random picking we get interval
    alpha = 0.05

    # get the quantiol of the student distribution (searches table), it gives it 1 - alpha/2 and n - 1 samples becouse it is in the formula 
    t_crit = st.t.ppf(1 - alpha/2, df=n_samples-1)

    # this impments the formula for the interval bounaries of boundary = avrage +- t_form_student_dis * (sqr(var / number_of_samples))
    ci_x = (mean_x_est - t_crit * np.sqrt(var_x_est / n_samples),
            mean_x_est + t_crit * np.sqrt(var_x_est / n_samples))
    ci_y = (mean_y_est - t_crit * np.sqrt(var_y_est / n_samples),
            mean_y_est + t_crit * np.sqrt(var_y_est / n_samples))

    conf_pct = (1 - alpha) * 100 

    print(f"\nInterval guess for avrage ({conf_pct:.0f}% confident interval):")
    print(f"IS for E[X]: ({ci_x[0]:.4f}, {ci_x[1]:.4f})")
    print(f"IS for E[Y]: ({ci_y[0]:.4f}, {ci_y[1]:.4f})")

