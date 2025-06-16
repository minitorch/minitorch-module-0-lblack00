"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul *
# - id *
# - add *
# - neg *
# - lt *
# - eq *
# - max *
# - is_close *
# - sigmoid *
# - relu *
# - log *
# - exp *
# - log_back *
# - inv *
# - inv_back *
# - relu_back *
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
	"""
	Result of multiplying x by y.

	Args:
		x: A float which is the multiplicand
		y: A float which is the multiplier

	Returns:
		Value of x * y
	"""
	return x * y

def id(a: float) -> float:
	"""
    Identity function.

    Args:
        a: A float input value

    Returns:
        The same value passed in
    """
	return a

def add(x: float, y: float) -> float:
	"""
	Result of adding x with y.

	Args:
		x: A float which is the first addend
		y: A float which is the second addend

	Returns:
		Sum of both addends
	"""
	return x + y

def neg(x: float) -> float:
	"""
	Negation of x.

	Args:
		x: A float which is to be negated

	Returns:
		Negation of x
	"""
	return -x

def lt(x: float, y: float) -> bool:
	"""
    Determine if x is less than y.

    Args:
        x: First float to compare
        y: Second float to compare

    Returns:
        True if x < y, False otherwise
    """
	return x < y

def eq(x: float, y: float) -> bool:
	"""
    Determine if x is equal to y.

    Args:
        x: First float to compare
        y: Second float to compare

    Returns:
        True if x equals y, False otherwise
    """
	return x == y

def max(x: float, y: float) -> float:
	"""
	Determine maximum value between x and y.

	Args:
		x: First float
		y: Second float

	Returns:
		Larger of the two values
	"""
	if x > y:
		return x

	return y

def is_close(x: float, y: float) -> bool:
	"""
    Check if two floats are approximately equal within a small tolerance.

    Args:
        x: First float
        y: Second float

    Returns:
        True if x and y are within 0.01 of each other
    """
	return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
	"""
    Compute sigmoid activation function.

    Args:
        x: Input float

    Returns:
        Sigmoid of x
    """
	if x >= 0:
		return 1.0 / (1.0 + math.e ** (-x))

	return (math.e ** x) / (1.0 + math.e ** x)

def relu(x: float) -> float:
	"""
    Compute ReLU (Rectified Linear Unit) activation.

    Args:
        x: Input float

    Returns:
        x if x > 0, else 0
    """
	if x <= 0.0:
		return 0.0

	return x

def log(x: float) -> float:
	"""
    Compute the natural logarithm of x.

    Args:
        x: Input float (must be > 0)

    Returns:
        Natural logarithm of x
    """
	return math.log(x)

def exp(x: float, y:float) -> float:
	"""
    Raise x to the power of y.

    Args:
        x: Base value
        y: Exponent value

    Returns:
        x raised to the power y
    """
	return x ** y

def inv(x: float) -> float:
	"""
	Find reciprocal of input value

	Args:
		x: Value to be inversed

	Returns:
		Reciprocal of x
	"""
	return 1.0 / x

def log_back(x: float, y: float) -> float:
	"""
    Compute the backward gradient of the log function.

    Args:
        x: Input to log
        y: Upstream gradient

    Returns:
        Gradient of log at x, scaled by upstream gradient y
    """
	return y * (1.0 / x) 

def inv_back(x: float, y: float) -> float:
	"""
    Compute the backward gradient of the inverse function.

    Args:
        x: Input to inv
        y: Upstream gradient

    Returns:
        Gradient of 1/x at x, scaled by upstream gradient y
    """
	return y * (1 / x**2)

def relu_back(x: float, y: float) -> float:
	"""
    Compute the backward gradient of the ReLU function.

    Args:
        x: Original input to ReLU
        y: Upstream gradient

    Returns:
        y if x > 0, else 0
    """
	if x < 0:
		return 0.0

	return y

# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
