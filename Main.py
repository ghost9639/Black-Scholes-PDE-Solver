"""Project file for MATH5350 Assessment 3"""

# Library Calls

from numba import njit         
import numpy as np
from scipy import linalg
from scipy.stats import norm
import time
from matplotlib import pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve


# @njit
def input_data (r, sigma, T, K, M, N, xmin, xmax):
    """Input validation function, returns false when input fails."""

    _fail = False               # fail condition allows for check and inverted result can be passed out


    if not (isinstance(r, (int, float)) and not isinstance(r, bool)) or r <= 0:
        print("The interest rate must be a positive real number.")
        _fail = True

    if not (isinstance(sigma, (int, float)) and not isinstance(sigma, bool)) or sigma <= 0:
        print("Volatility must be a positive real number.")
        _fail = True

    if not (isinstance(T, (int, float)) and not isinstance(T, bool)) or T <= 0:
        print("Time to maturity must be a positive real number.")
        _fail = True

    if not (isinstance(K, (int, float)) and not isinstance(K, bool)) or K <= 0:
        print("Strike price must be a positive real number.")
        _fail = True

    if not (isinstance(M, int) or isinstance(M, np.integer)) or isinstance(M, bool) or M <= 0:
        print("M must be a positive integer.")
        _fail = True

    if not (isinstance(N, int) or isinstance(N, np.integer)) or isinstance(N, bool) or N <= 0:
        print("N must be a positive integer.")
        _fail = True

    if not (isinstance(xmin, (int, float)) and not isinstance(xmin, bool)):
        print("xmin must be a real number.")
        _fail = True

    if not (isinstance(xmax, (int, float)) and not isinstance(xmax, bool)):
        print("xmax must be a real number.")
        _fail = True

    if isinstance(xmin, (int, float)) and isinstance(xmax, (int, float)) and xmin >= xmax:
        print("xmin must be strictly less than xmax.")
        _fail = True
    
    return not _fail


# numba used here to accelerate required for loops
@njit(fastmath=True)
def tridiagonal_solve_thomas (alpha, beta, gamma, b):
    """Thomas algorithm for solving Ax = b with respect to x,

    where A is an n-dimensional square tridiagonal matrix, b is a vector of values length n, and x
    is a vector of values length n.
    
    Expects alpha as the diagonal vector of A, beta as the superdiagonal of A, and gamma as the subdiagonal."""
    
    n = len(b)                  # length of longest diagonal (or dim of matrix)

    # defining our transformed diagonals
    hat_alpha = np.zeros(n, dtype = np.float64) # float64 required for njit efficiency
    hat_b = np.zeros(n, dtype = np.float64)

    # Initial conditions for BTCS loop
    hat_alpha[0] = alpha[0]
    hat_b[0] = b[0]

    # Conversion to simple matrix terms
    for i in range (1, n):
        hat_alpha[i] = alpha[i] - ((beta[i-1] * gamma[i-1]) / hat_alpha[i-1])
        hat_b[i] = b[i] - ((hat_b[i-1] * gamma[i-1]) / hat_alpha[i-1])

    # Solution vector simple calculation
    x = hat_b / hat_alpha
    
    for i in range (n-2, -1, -1):
        x[i] = (hat_b[i] - beta[i] * x[i+1]) / hat_alpha[i]
        
    return x


def thomas_columns (A):
    """Converts a square tridiagonal array into the required Thomas algorithm vectors. Experts a tridiagonal array,
    returns the diagonal, super diagonal, and sub-diagonal as numpy arrays (of numpy.float64s)."""

    n = len(A)
    sub_diag = np.zeros(n-1, dtype = np.float64)
    super_diag = np.zeros(n-1, dtype = np.float64)
    diag = np.zeros(n, dtype = np.float64)

    for i in range (n):         # iterates through central diagonal
        diag[i] = A[i][i]

    for i in range (n-1):
        super_diag[i] = A[i][i+1] # catches value to right of central diagonal

    for i in range (1, n):
        sub_diag[i-1] = A[i][i-1] # catches value to left of central diagonal

    # deep copy is memory inefficient, only use this within other function calls, otherwise create an index to call
    return diag, super_diag, sub_diag


def generate_banded (A, extra_diags): 
    """Utility for generating banded ab matrix for scipy.

    linalg.solve_banded requires a banded matrix of the format ab[u+i-j,j] = A[i,j], where A is the diagonal
    matrix in question (not necessarily tridiagonal). The extra_diags argument is the number of diagonals
    below and above the main diagonal, for a tridiagonal matrix this is simply (1,1).
    Returns the banded matrix as required."""

    n = len(A)
    l, u = extra_diags
    
    banded = np.zeros((l+u+1, n))
    
    for i in range (n):
        for j in range (n):

            if 0 <= u+i-j < l+u+1:
                banded[u+i-j, j] = A[i,j]

    return banded

# We can also compare the speeds of both solvers, accounting for the fact that njit compiles on the first
# function call (taking less time on successive ones), and the banded matrix must be generated for scipy
# to begin with

def generate_tridiagonal_matrix (n, low = 0, high = 10):
    """Utility for sample tridiagonal matrix generation. Expects an integer n, and a tuple range, from low to high,
    and returns a tridiagonal matrix of dimensions n*n. """

    try:
        assert isinstance(n, int)
    except AssertionError:
        print("n must be an integer.")
        return None
    
    mat = np.zeros((n,n), dtype = np.float64)
    # low, high = range

    for i in range (n):
        for j in range (n):

            # central diag
            if i == j:
                mat[i][j] = np.random.uniform(low, high)

            # sub diag
            elif i == j-1:
                mat[i][j] = np.random.uniform(low, high)

            # super diag
            elif i == j+1:
                mat[i][j] = np.random.uniform(low, high)

    return mat

def testing_thomas_accuracy ():
    """Utility for testing whether Thomas algorithm is accurate. Robust against float precision errors."""

    # defining performance testing variables
    N = [100, 1000, 5000, 10000]
    n_trials = 3

    for size in N:

        print(f"Running tests on size {size}...")
        # required formats for both functions
        test_mat = generate_tridiagonal_matrix(size) # actual matrix
        test_alpha, test_beta, test_gamma = thomas_columns(test_mat) # diagonals
        test_banded = generate_banded(test_mat, (1,1)) # scipy banded matrix

        for t in range (n_trials):

            test_b = np.random.uniform(0.0, 10.0, size)    # RHS matrix

            thomas_solutions = tridiagonal_solve_thomas (test_alpha, test_beta, test_gamma, test_b)
            scipy_solutions = linalg.solve_banded((1,1), test_banded, test_b)

            for i in range (t-1):

                try:
                    assert np.isclose(thomas_solutions[i], scipy_solutions[i], rtol = 1e-12, atol = 1e-15)
                except AssertionError:
                    print(f"Error on trial {size}.")
                    return False

    print("All tests successful.")
    return True

def timing_thomas_solvers ():
    """Performance testing for linalg.solve_banded and tridiagonal_solve_thomas.
    
    Requires a tridiagonal matrix A of dimensions n by n, and vector b of length n.
    Returns the required times for both functions to return a vector x of length n such that Ax = b."""

    # numba compilation step call, not tested
    tridiagonal_solve_thomas(np.array([3,4,10]), np.array([6,2]), np.array([9,7]), np.array([10, 6, 2]))

    # defining performance testing variables
    N = [100, 1000, 5000, 10000, 30000]
    thomas_times = []
    scipy_times = []
    n_trials = 5

    for size in N:

        # required formats for both functions
        test_mat = generate_tridiagonal_matrix(size) # actual matrix
        test_alpha, test_beta, test_gamma = thomas_columns(test_mat) # diagonals
        test_banded = generate_banded(test_mat, (1,1)) # scipy banded matrix

        thomas_trial_times = []
        scipy_trial_times = []

        for i in range (n_trials):

            print(f"Currently testing matrices of size {size}: Trial {i}...")

            test_b = np.random.uniform(0.0, 10.0, size)    # RHS matrix

            # Timing tridiagonal Thomas algorithm
            start_time = time.perf_counter_ns()
            tridiagonal_solve_thomas (test_alpha, test_beta, test_gamma, test_b)
            end_time = time.perf_counter_ns()
            
            thomas_trial_times.append(end_time - start_time)

            # Timing scipy algorithm
            start_time = time.perf_counter_ns()
            linalg.solve_banded((1,1), test_banded, test_b)
            end_time = time.perf_counter_ns()
        
            scipy_trial_times.append(end_time - start_time)

        thomas_times.append(np.mean(thomas_trial_times))
        scipy_times.append(np.mean(scipy_trial_times))

    return thomas_times, scipy_times, N

# Question 3 =============================================
################ The Actual Function #####################
@njit(fastmath=True)
def black_scholes_implicit_engine (r, sigma, T, K, M, N, xmin, xmax):
    """Implicit calculation for European Call option

    h(S_T) = (S_T-K)^+
    Uses the Implicit method and returns a vector of the stock evolution.
    @param r the interest rate (as a decimal)
    @param sigma the volatility of the stock
    @param T the time till maturity (in years)
    @param K the strike price
    @param M the number of splits in the stock
    @param N the number of splits in time
    @param xmin the lower domain of the stock
    @param xmax the upper domain of the stock
    @output w the approximations of the stock price
    """

    # domains
    domain_x = np.linspace(xmin, xmax, M+1) # x is the value of the stock
    time = np.linspace(0, T, N+1)           # time is the "location" of the split (between 0 and T)
    
    # Discretisation constants
    dt = T / N
    
    # Call price evolution (effectively range)
    w = np.zeros((M+1,N+1), dtype = np.float64)

    # Call boundary conditions
    w[:,0] = np.maximum (domain_x - K, 0) # value of x
    w[0,:] = 0                  # when x -> 0
    w[-1,:] = xmax - (K * np.exp(-r * time)) # when x -> \infty
    
    # A_implicit matrix definition ========

    # this is the discretised space for defining row column vectors across stock value
    space = np.arange(0, M+1, dtype = np.float64)

    # instead of storing whole matrix define diagonal vectors
    alpha = 0.5 * dt * ((sigma**2) * (space**2) - r * space) # upper diag
    beta = dt * ((sigma**2) * (space**2) + r)       # middle diag
    gamma = 0.5 * dt * ((sigma**2) * (space**2) + r * space) # lower diag

    # impliciT method A Matrix
    T_lower = -gamma[1:-2]
    T_main = 1 + beta[1:-1]
    T_upper = -alpha[2:-1]


    for v in range (1, N+1):         # across timesteps
        
        # boundary conditions
        RHS = w[1:M,v-1].copy() # need a deep copy of last step

        # addition of boundary conditions (RHS is now w_{v-1} + d_v)
        RHS[0] += -alpha[1] * (w[0, v])
        RHS[-1] += gamma[M - 1] * (w[M,v])
        
        # Solve for current period step (since Aw_{v+1} = (w_v + d_v))
        w[1:M, v] = tridiagonal_solve_thomas(T_main, T_lower, T_upper, RHS)

    return w

def black_scholes_implicit (r, sigma, T, K, M, N, xmin, xmax):

    try:
        assert input_data (r, sigma, T, K, M, N, xmin, xmax)
    except AssertionError:
        print("Please adjust listed inputs.")
        return None

    return black_scholes_implicit_engine (r, sigma, T, K, M, N, xmin, xmax)

def black_scholes_exact (S_0, r, sigma, T, K):
    """A helper utility that explicitly calculates the Black-Scholes price for a European call

    @param S_0 the initial stock price
    @param r the interest rate (as a decimal)
    @param sigma the volatility
    @param T the time till maturity (in years)
    @parma K the strike price (same units as stock price)"""

    d_1 = (np.log(S_0 / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)

    return S_0 * norm.cdf(d_1) - np.exp(-r * T) * K * norm.cdf(d_2)

def testing_implicit_accuracy (r, sigma, T, K, M, N, xmin, xmax):
    """Generates the price evolution from the explicit Black-Scholes and the implicit method
    across the same stock and price domains and ranges holomorphically

    @param r the interest rate (as a decimal)
    @param sigma the volatility of the stock
    @param T the time till maturity (in years)
    @param K the strike price
    @param M the number of splits in the stock
    @param N the number of splits in time
    @param xmin the lower domain of the stock
    @param xmax the upper domain of the stock
    @output exact_solution the exact price as calculated by the Black-Scholes equation
    @output w the implicit approximations of the stock price
    @output S_domain the stock price domain
    """

    try:
        assert input_data (r, sigma, T, K, M, N, xmin, xmax)
    except AssertionError:
        print("Please adjust listed inputs.")
        return None
    # The implicit function run in full (to preserve the stock price split) ========
    
    # Approximation constants
    delta_tau = ((sigma ** 2) * T) / (2 * N)
    delta_x = (xmax - xmin) / M
    
    llambda = delta_tau / (delta_x ** 2)

    # Approximation evolution
    w = np.zeros((M+1,N+1), dtype = np.float64)
    

    # initial period value
    for i in range (M+1):
        x = xmin + i * delta_x
        w[i][0] = max(0., np.exp(x) - K) # * np.exp(-r * T) 
    

    # central differenced matrix
    alpha = (1 + 2 * llambda) * np.ones(M-1)
    beta = (-llambda) * np.ones(M-2)
    gamma = (-llambda) * np.ones(M-2)
    
    
    for v in range (N):

        tau = (v+1) * delta_tau # current time step for discounting
        
        # boundary conditions        
        w[0, v+1] = 0.
        w[M, v+1] = np.exp(xmax) - (K * np.exp((-2 * r * tau) / (sigma ** 2)))

        # w_v vector (copy of last chain)
        RHS = w[1:M,v].copy()

        # addition of boundary conditions (RHS is now w_v + d_v)
        RHS[0] += llambda * w[0, v+1]  # left
        RHS[-1] += llambda * w[M, v+1] # right
        
        # Solve for current period step (since Aw_{v+1} = (w_v + d_v))
        w[1:M, v+1] = tridiagonal_solve_thomas(alpha, beta, gamma, RHS)

    # exact solutions ========

    x_domain = xmin + np.arange(M+1) * delta_x

    S_domain = np.exp(x_domain) # (x = log(S))

    exact_solution = black_scholes_exact(S_domain, r, sigma, T, K)

    # We can now return the exact solution and the central differenced approximation
    return exact_solution, w, S_domain

# Question 4 =============================================
################ The Actual Function #####################

@njit(fastmath=True)
def _stancil_algorithm (lower, central, upper, w):
    """helper function that implements the Stancil tridiagonal vector multiplication algorithm"""

    n = w.shape[0]
    y = np.empty(n, dtype=np.float64)

    for i in range (n):
        val = central[i] * w[i]
        if i > 0:
            val += lower[i-1] * w[i-1]
        if i < n-1:
            val += upper[i] * w[i+1]
        y[i] = val
    return y


@njit(fastmath=True)
def black_scholes_crank_nicolson_engine (r, sigma, T, K, M, N, xmin, xmax):
    """Calculates and returns the price of a vanilla European Call option using the Crank Nicolson
    approximation

    @param r the interest rate (as a decimal)
    @param sigma the volatility of the stock
    @param T the time till maturity (in years)
    @param K the strike price
    @param M the number of splits in the stock
    @param N the number of splits in time
    @param xmin the lower domain of the stock
    @param xmax the upper domain of the stock
    @output w the approximations of the stock price"""

    # domains
    domain_x = np.linspace(xmin, xmax, M+1) # x is the value of the stock
    time = np.linspace(0, T, N+1)           # time is the "location" of the split (between 0 and T)

    # Discretisation constants
    dt = T / N

    # Call price evolution (effectively range)
    w = np.zeros((M+1,N+1)) # w holds the call price for (stock value, time split)

    # Call boundary conditions
    w[:,0] = np.maximum (domain_x - K, 0) # value of x
    
    w[0,:] = 0                  # when x -> 0
    w[-1,:] = xmax - (K * np.exp(-r * time)) # when x -> \infty
    
    # Crank Nicolson Scheme ==============
    
    space = np.arange(0, M+1)   # this is the discretised space for defining row column vectors across stock value
    
    alpha = 0.25 * dt * ((sigma**2) * (space**2) - r * space) # upper diag
    beta = -0.5 * dt * ((sigma**2) * (space**2) + r)       # middle diag
    gamma = 0.25 * dt * ((sigma**2) * (space**2) + r * space) # lower diag
    
    # explicit method A matrix
    # A_exp = sparse.diags([alpha[2:], 1+beta[1:], gamma[1:]], [-1,0,1], shape=(M-1, M-1))

    A_lower = alpha[2:]
    A_central = beta[1:] + 1
    A_upper = gamma[1:-1]

    # implicit method A Matrix
    T_lower = -gamma[1:-1]
    T_central = 1 - beta[1:]
    T_upper = -alpha[2:]

    # iterations
    for v in range (1, N+1):
        
        # calculate Bn * w_v
        b = _stancil_algorithm(A_lower, A_central, A_upper, w[1:-1, v-1])
        
        # add boundary conditions
        b[0] += alpha[1] * (w[0, v-1] + w[0, v]) 
        b[-1] += gamma[M - 1] * (w[M, v] + w[M, v-1])

        # solve for next price step
        w[1:M, v] = tridiagonal_solve_thomas(T_central, T_lower, T_upper, b)
        
    return w

def black_scholes_crank_nicolson (r, sigma, T, K, M, N, xmin, xmax):

    try:
        assert input_data (r, sigma, T, K, M, N, xmin, xmax)
    except AssertionError:
        print("Please adjust listed inputs.")
        return -1

    return black_scholes_crank_nicolson_engine (r, sigma, T, K, M, N, xmin, xmax)

    


def _rmse(approx, exact):
    return np.sqrt(np.mean((approx - exact) ** 2))


def testing_rmse (r = 0.05, sigma = 0.1, T = 1.8, K = 673, xmin = 168.25, xmax = 2692):
    """Utility for calculating the root mean squared error for the implicit and Crank_Nicolson schemes

    @param r the interest rate as a decimal
    @param sigma the volatility
    @param T the maturity (in years)
    @param K the strike price
    @output
    """

    try:
        assert input_data (r, sigma, T, K, 10, 10, -5, 5)
    except AssertionError:
        print("Please adjust listed inputs.")
        return None

    M_cases = [10, 30, 50, 100, 250, 1000]
    N_cases = [10, 30, 50, 100, 250, 1000]
    
    implicit_rmse       = np.zeros((len(M_cases), len(N_cases)), dtype=np.float64)
    crank_nicolson_rmse = np.zeros((len(M_cases), len(N_cases)), dtype=np.float64)

    for i, M in enumerate(M_cases):
        for j, N in enumerate(N_cases):

                num = black_scholes_implicit (r, sigma, T, K,
                                                   M, N, xmin, xmax)
                nic_num = black_scholes_crank_nicolson (r, sigma, T, K, M, N, xmin, xmax)
                mesh = np.linspace(xmin, xmax, M+1)
                exact = black_scholes_exact(mesh, r, sigma, T, K)

            
                abs_error = np.abs(num[:,-1] - exact)
                rmse = np.sqrt(np.mean(abs_error**2))

    
                abs_error_nic = np.abs(nic_num[:,-1] - exact)
                rmse_nic = np.sqrt(np.mean(abs_error_nic**2))

                implicit_rmse[i,j] = rmse
                crank_nicolson_rmse[i,j] = rmse_nic

    return M_cases, N_cases, implicit_rmse, crank_nicolson_rmse
    

            
if __name__ == "__main__":

    # Question 2 tridiagonal_solve_thomas function ================================
    
    # This will serve as a simple example case for the Thomas solver
    example_mat = np.array([
        [3, 6, 0, 0],
        [9, 4, 2, 0],
        [0, 7, 10, 8],
        [0, 0, 12, 4],
    ])

    example_sol = np.array([10, 6, 2, 4])

    print(thomas_columns(example_mat)[1])

    print(tridiagonal_solve_thomas(np.array(thomas_columns(example_mat)[0]), np.array(thomas_columns(example_mat)[1]),
                                   np.array(thomas_columns(example_mat)[2]), example_sol))
    # This gave me: array([-0.49084249,  1.91208791,  1.38461538, -3.15384615])

    banded_example = generate_banded(example_mat, (1,1)) # just a utility that formats the matrix for scipy

    # This also gives the solutions array([-0.49084249,  1.91208791,  1.38461538, -3.15384615])
    print(linalg.solve_banded((1,1), banded_example, example_sol))

    # helper that tests whether tridiagonal_solve_thomas and generate_banded find same value within
    # floating point tolerance
    testing_thomas_accuracy()

    
    # helper that checks runtime
    t_times, s_times, nums = timing_thomas_solvers()
    nums = np.array(nums, dtype = np.float64)
    
    c_linear = t_times[0] / nums[0]
    linear_ref = c_linear * nums

    c_quad = t_times[0] / nums[0]**2
    quad_ref = c_quad * nums**2

    plt.loglog(nums, linear_ref, linestyle = '--', label = r"$O(n)$ reference")
    plt.loglog(nums, quad_ref, linestyle = ':', label = r"$O(n^2)$ reference")

    plt.loglog(nums, t_times, marker = 'o', label = "Thomas algorithm")
    plt.loglog(nums, s_times, marker = 'x', label = "scipy banded algorithm")
    
    plt.title("Comparison of scipy function and Thomas algorithm")
    plt.xlabel("Sample matrix size")
    plt.ylabel("Runtime (logged nanoseconds)")
    plt.legend()
    plt.show()


    # Question 3 the implicit approximation =====================
    
    # our test values
    test_r = 0.05
    test_sigma = 0.1
    test_T = 1.8
    test_K = 673
    test_M = 4
    test_N = 3
    
    test_xmin = test_K/4 # this range is a good distance above and below the strike price
    test_xmax = test_K*4 # K/2 and K*3 are also similar ranges I've seen

    test_num = black_scholes_implicit (test_r, test_sigma, test_T, test_K, test_M, test_N, test_xmin, test_xmax)
    test_nic_num = black_scholes_crank_nicolson (test_r, test_sigma, test_T, test_K, test_M, test_N, test_xmin, test_xmax)
    test_mesh = np.linspace(test_xmin, test_xmax, test_M+1)
    test_exact = black_scholes_exact(test_mesh, test_r, test_sigma, test_T, test_K)

    plt.title(f"Option Price Estimates for {test_N} timesplits and {test_M} stock splits")
    plt.xlabel(f"Stock Value split (between S=({test_xmin:.2f}, {test_xmax:.2f}))")
    plt.ylabel("Option Price")

    plt.plot(test_exact, ":", label="Exact")

    plt.plot(test_num[:,-1], "--", label="Implicit First Difference (Final Run)")
    # plt.plot(test_nic_num[:,-1], "-.", label="Crank Nicolson Method (Final Run)")

    plt.plot(test_num[:,1], "--", label="Implicit First Difference (First Run)")
    # plt.plot(test_nic_num[:,1], "-.", label="Crank Nicolson Method (First Run)")

    # plt.xlim(11.85,12.04)
    # plt.ylim(2050, 2080)

    plt.legend()
    plt.show()

    print(f"The final exact value is £{test_exact[-1]:.2f}, "
          f"the initial implicit estimate is £{test_num[-1,0]:.2f}, "
          f"and the final implicit estimate is £{test_num[-1,-1]:.2f}.")

    print(f"The final exact value is £{test_exact[-1]:.2f}, "
          f"the initial Crank Nicolson estimate is £{test_nic_num[-1,0]:.2f}, "
          f"and the final Crank Nicolson estimate is £{test_nic_num[-1,-1]:.2f}.")


    # manual errors
    abs_error = np.abs(test_num[:,-1] - test_exact)

    max_error = np.max(abs_error)
    
    rmse = np.sqrt(np.mean(abs_error**2))

    
    abs_error_nic = np.abs(test_nic_num[:,-1] - test_exact)

    max_error_nic = np.max(abs_error_nic)
    rmse_nic = np.sqrt(np.mean(abs_error_nic**2))

    print(f"The base model RMSE is £{rmse:.2f}, and the maximum error is £{max_error:.2f}")    
    print(f"The Crank Nicolson RMSE is £{rmse_nic:.2f}, and the maximum error is £{max_error_nic:.2f}")
    rmse - rmse_nic


    M_cases, N_cases, rmse_implicit, rmse_crank = testing_rmse()



    im = plt.imshow(
        rmse_implicit,
        origin="lower",
        aspect="auto",
        extent=[
            N_cases[0],
            N_cases[-1],
            M_cases[0],
            M_cases[-1]
        ]
    )

    plt.colorbar(im, label="RMSE")

    plt.xlabel("N (time discretisation)")
    plt.ylabel("M (space discretisation)")

    plt.title("Implicit Scheme RMSE Heatmap")

    plt.show()

    
    im = plt.imshow(
        rmse_crank,
        origin="lower",
        aspect="auto",
        extent=[
            N_cases[0],
            N_cases[-1],
            M_cases[0],
            M_cases[-1]
        ]
    )

    plt.colorbar(im, label="RMSE")

    plt.xlabel("N (time discretisation)")
    plt.ylabel("M (space discretisation)")

    plt.title("Crank Nicolson RMSE Heatmap")

    plt.show()


    rmse_implicit
    rmse_crank
    
