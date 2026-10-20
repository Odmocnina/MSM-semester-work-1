from typing import cast

import matplotlib.pyplot as plt
import numpy as np
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

if __name__ == "__main__":

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

    # =================================================================================================================

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

    # =================================================================================================================

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

    print(f"E[X] = {E_x} ≈ {float(E_x):.4f}")
    print(f"E[Y] = {E_y} ≈ {float(E_y):.4f}")
    print(f"Var[X] = {var_x} ≈ {float(var_x):.5f}")
    print(f"Var[Y] = {var_y} ≈ {float(var_y):.5f}")
    print(f"cov(X,Y) = {cov_xy} ≈ {float(cov_xy):.5f}")
    print("Variation matrix:")
    sp.pprint(cov_matrix)
