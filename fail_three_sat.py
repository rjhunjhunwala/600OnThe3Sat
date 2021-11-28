import random
import pandas as pd
import collections
import time
import pysmt
from pysmt.shortcuts import Symbol, LE, GE, Int, And, Equals, Plus, Solver, Or, Iff, Bool, get_model
from pysmt.typing import INT
from mip import *
from functools import reduce
from itertools import combinations
from operator import mul
from scipy.optimize import least_squares
from hyperopt import fmin, tpe, space_eval, hp

critical_ratio = 4.4
REPEATS = 5
TIMEOUT = 10
MIN_N = 3
MAX_N = 100

# https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/descr.html#:~:text=One%20particularly%20interesting%20property%20of%20uniform%20Random-3-SAT%20is,systematically%20increasing%20%28or%20decreasing%29%20the%20number%20of%20kclauses
# We vigorously handwave the phase transition for 3sat


"""
Benchmark various free ways to solve 3sat
"""

def create_random_ksat(num_variables, num_clauses, k = 3):
    """
    Return a random 3sat clause with num_variable  number of variables and num_clauses clauses

    :param num_variables:
    :param num_clauses:
    :param k: k in ksat
    :return: A list of K-tlists of variables. Each variable is tuple contains a pair, which is an integer (the name of the variable)
    and whether or not it is negated. This is a 3sat clause,in CnF
    """
    def valid(clause):
        return len(set(var for var, _ in clause)) == len(clause)
    def create_clause():
        while True:
            clause = tuple((random.choice(range(num_variables)), random.random() < .5) for i in range(k))
            if valid(clause):
                return clause

    clauses = set()
    while len(clauses) < num_clauses:
        new_clause = create_clause()
        while new_clause in clauses:
            new_clause = create_clause()
        clauses.add(new_clause)

    return list(clauses)

def evaluate(cnf, variables):
    return all(any(variables[name] == val for name, val in clause) for clause in cnf)

def get_num_symbols(sat_instance):
    return max(max(tup[0] for tup in clause) for clause in sat_instance) + 1

def canonical_solver(sat_instance):
    """
    Reference solver. Assume complete and sound.
    :param sat_instance:
    :return:
    """

    num_symbols = get_num_symbols(sat_instance)
    symbols = [Symbol(str(i)) for i in range(num_symbols)]
    domains = [Or([Iff(Bool(is_true), symbols[variable]) for variable, is_true in clause]) for clause in sat_instance]

    formula = And(domains)

    model = get_model(formula)
    if model:
        return True
    else:
        return False

def assignment_from_num(i, num):
    return [bool((i >> index) & 1) for index in range(num)]

def nonconvex_local(sat_instance):
    n = get_num_symbols(sat_instance)

    def cost(x):
        return [sum(
            min(int(1 - x[variable]) if is_true else int(x[variable]) for variable, is_true in clause)
            for clause in sat_instance
        )
        ]

    results = []

    for i in range(10):
        start = [int(random.random() < .5) for i in range(n)]
        result = least_squares(cost, start, bounds = ([0] * n, [1] * n))

        guessed_output = [int(a >= 0.5) for a in result.x]

        results.append(evaluate(sat_instance, guessed_output))

    return any(results)

def hyperopt(sat_instance):
    n = get_num_symbols(sat_instance)
    def cost(x):
        return sum([
            min(int(1 - x[variable]) if is_true else int(x[variable]) for variable, is_true in clause)
            for clause in sat_instance
        ])
    c = 2.1
    best = fmin(fn=cost,
                space=[hp.randint('x' + str(i), 0, 2) for i in range(n)],
                algo=tpe.suggest,
                max_evals = 2 * int(n ** c))

    return cost(list(best.values())) < 1

def brute_force(sat_instance):
    """
    Solve in Exponential time. For fun. O(c * (2 ** n))
    :param sat_instance:
    :return:
    """
    def all_instances(num):
        for i in range(2 ** num):
            yield assignment_from_num(i, num)
    return any(evaluate(sat_instance, s) for s in all_instances(get_num_symbols(sat_instance)))

def do_benchmark() -> pd.DataFrame:
    solution_strategies = {"canonical":canonical_solver, "ilp": do_cbc_solver, "schonig": schonig,
                           "crank_algorithm": crank_algorithm, "local_sat": local_sat,
                           "brute_force": brute_force, "nonconvex_local": nonconvex_local, "hyperopt": hyperopt}
    hit_cutoffs = set()
    ns = [int(a / REPEATS) for a in range(MIN_N * REPEATS, MAX_N * REPEATS, 1)]
    cols = collections.defaultdict(list)

    for n in ns:
        new_row = dict()
        new_row["n"] = n
        instance = create_random_ksat(n, int(n * critical_ratio))
        for solution_name, solution in solution_strategies.items():
            start = time.time()
            new_row[solution_name] = solution(instance) if solution_name not in hit_cutoffs else False
            end = time.time()
            new_time = end - start
            new_row[solution_name + "_time"] = new_time
            if new_time > TIMEOUT:
                hit_cutoffs.add(solution_name)
        right_solution = new_row["canonical"]

        for solution_name in solution_strategies:
            if solution_name in hit_cutoffs:
                new_row[solution_name + "_correct"] = False
            else:
                new_row[solution_name + "_correct"] = right_solution == new_row[solution_name]

        for key in new_row:
            cols[key].append(new_row[key])

    return pd.DataFrame(cols)

def do_cbc_solver(sat_instance):
    n = get_num_symbols(sat_instance)

    m = Model("knapsack", solver_name = CBC)

    x = [m.add_var(var_type=BINARY) for i in range(n)]

    for clause in sat_instance:
        m += xsum(x[var] if is_true else 1 - x[var] for var, is_true in clause) >= 1

    status = m.optimize()

    return status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE


def schonig(sat_instance):
    """
    Schonig's algorithm
    :param sat_instance:
    :return:
    """
    n = get_num_symbols(sat_instance)

    def attempt_greedy_walk():
        randomized_assignment = [random.random() < .5 for i in range(n)]
        c = len(sat_instance)
        for i in range(5 * c):
            evaluation = evaluate(sat_instance, randomized_assignment)
            if evaluation:
                return True
            for clause in sat_instance:
                if not evaluate([clause], randomized_assignment):
                    var, _ = random.choice(clause)
                    randomized_assignment[var] = not randomized_assignment[var]
                    break

        return False


    return any(attempt_greedy_walk() for i in range(10))

def local_sat(sat_instance):
    """
    Gradient descent esque sat, with some simulated annealing. Should be worse than schonig better better than the crank
    :return:
    """
    """
    Schonig's algorithm
    :param sat_instance:
    :return:
    """
    n = get_num_symbols(sat_instance)

    map = [0] * n

    for clause in sat_instance:
        for variable, is_true in clause:
            map[variable] += 1 - (2 * is_true)

    def attempt_greedy_walk():
        randomized_assignment = [random.random() < .5 for i in range(n)]
        c = len(sat_instance)
        for i in range(5 * c):
            evaluation = evaluate(sat_instance, randomized_assignment)
            if evaluation:
                return True
            for clause in sat_instance:
                if not evaluate([clause], randomized_assignment):
                    var, _ = max(clause, key = lambda tup: ((map[tup[0]] if not randomized_assignment[tup[0]] else -map[tup[0]]), random.random()))
                    randomized_assignment[var] = not randomized_assignment[var]
                    break

        return False


    return any(attempt_greedy_walk() for i in range(30))


def crank_algorithm(sat_instance):
    """
    If this works the following author is a millionare, and P = BPP
    https://arxiv.org/ftp/arxiv/papers/1703/1703.01905.pdf
    :param sat_instance:
    :return:
    """

    n = get_num_symbols(sat_instance)
    # M is some free parameter less than n, lets fix arbitrarily
    M = n - 1
    # For some reason M is assumed to be even
    if M % 2:
        M = M - 1

    M = 4

    current_assignment = [int(M / 2) for i in range(n)]

    def evaluate_fractional_clause(clause, variables):

        k = len(clause)
        out = 0
        for subset in range(1, k + 1):
            mult = (-1) ** (subset + 1)
            for combo in combinations(range(k), subset):

                out += reduce(mul,(((variables[clause[i][0]]) if clause[i][1] else (M - (variables[clause[i][0]] / M))) for i in combo)) * mult / (M ** subset)

        return out

    def worst_clause_and_val():
        return min(((clause, evaluate_fractional_clause(clause, current_assignment)) for clause in sat_instance), key = lambda a: (a[1], random.random()))

    for i in range(20 * n * n * M * M):
        assert all(var <= M for var in current_assignment)
        worst_clause, worst_clause_truth_value = worst_clause_and_val()
        if worst_clause_truth_value == 1:
            return True
        else:
            random_var, _ = random.choice(worst_clause)
            increments = {0.0: [1], M: [-1]}
            increment_choice = random.choice(increments.get(current_assignment[random_var], [1, -1]))
            current_assignment[random_var] += increment_choice

    return False



benchmark_df = do_benchmark()

# print(do_cbc_solver(create_random_ksat(10, 100)))

benchmark_df.to_csv("data", index = False)
benchmark_df.groupby("n").mean().to_csv("data_grouped", index = True)

pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", 1000)
print(benchmark_df)
print(benchmark_df.groupby("n").mean())

# print(benchmark_df)