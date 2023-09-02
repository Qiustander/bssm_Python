import numpy as np


def assertAllInRange(target,
                     lower_bound,
                     upper_bound,
                     open_lower_bound=False,
                     open_upper_bound=False):
    """Assert that elements in a Tensor are all in a given range.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      lower_bound: lower bound of the range
      upper_bound: upper bound of the range
      open_lower_bound: (`bool`) whether the lower bound is open (i.e., > rather
        than the default >=)
      open_upper_bound: (`bool`) whether the upper bound is open (i.e., < rather
        than the default <=)

    Raises:
      AssertionError:
        if the value tensor does not have an ordered numeric type (float* or
          int*), or
        if there are nan values, or
        if any of the elements do not fall in the specified range.
    """
    target = np.array(target)
    if not (np.issubdtype(target.dtype, np.floating) or
            np.issubdtype(target.dtype, np.integer)):
        raise AssertionError(
            'The value of %s does not have an ordered numeric type, instead it '
            'has type: %s' % (target, target.dtype))

    nan_subscripts = np.where(np.isnan(target))
    if np.size(nan_subscripts):
        raise AssertionError(
            '%d of the %d element(s) are NaN. '
            'Subscripts(s) and value(s) of the NaN element(s):\n' %
            (len(nan_subscripts[0]), np.size(target)) )

    range_str = (('(' if open_lower_bound else '[') + str(lower_bound) + ', ' +
                 str(upper_bound) + (')' if open_upper_bound else ']'))

    violations = (
        np.less_equal(target, lower_bound) if open_lower_bound else np.less(
            target, lower_bound))
    violations = np.logical_or(
        violations,
        np.greater_equal(target, upper_bound)
        if open_upper_bound else np.greater(target, upper_bound))
    violation_subscripts = np.where(violations)
    if np.size(violation_subscripts):
        raise AssertionError(
            '%d of the %d element(s) are outside the range %s. ' %
            (len(violation_subscripts[0]), np.size(target), range_str) +
            'Subscript(s) and value(s) of the offending elements:\n')
