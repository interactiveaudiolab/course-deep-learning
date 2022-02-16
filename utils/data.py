import numpy as np
from numpy import pi


################################################################################
# Datasets
################################################################################


def make_two_gaussians_data(
        examples_per_class: int,
        distance_between_means: float
):
    """
    Create a 2-dimensional set of points, where half the points are drawn from
    one Gaussian distribution and the other half are drawn from a different Gaussian

    PARAMETERS
    ----------
    examples_per_class      An integer determining how much data we'll generate

    distance_between_means  Distance between the means of the two Gaussians.

    RETURNS
    -------
    data      A numpy array of 2 columns (dimensions) and 2*examples_per_class rows

    labels    A numpy vector with 2*examples_per_class, with a +1 or -1 in each
              element. The jth element is the label of the jth example"""

    mean = [0, 0]
    cov = [[1, 0], [0, 1]]

    # negative class -1
    negData = np.random.multivariate_normal(mean, cov, examples_per_class)

    # positive class 1
    posData = np.random.multivariate_normal(mean, cov, examples_per_class)
    posData += distance_between_means

    # make the labels
    negL = np.ones(examples_per_class) * -1
    posL = np.ones(examples_per_class)

    # wrap it up and ship it out!
    data = np.concatenate([posData, negData])
    labels = np.concatenate([posL, negL])
    
    # shuffle the data
    perm = np.random.permutation(len(labels))
    data = data[perm]
    labels = labels[perm]

    return data, labels


def make_XOR_data(examples_per_class: int):
    """
    Create a 2-dimensional set of points in the XOR pattern. Things in the
    upper right and lower left quadrant are class 1. Things in the other two
    quadrants are class -1.

    PARAMETERS
    ----------
    examples_per_class      An integer determining how much data we'll generate

    RETURNS
    -------
    data      A numpy array of 2 columns (dimensions) and 2*examples_per_class rows

    labels    A numpy vector with 2*examples_per_class, with a +1 or -1 in each
              element. The jth element is the label of the jth example"""

    mean = [0, 0]
    cov = [[1, 0], [0, 1]]

    # make a circular unit Gaussian and sample from it
    data = np.random.multivariate_normal(mean, cov, examples_per_class*2)

    x = data.T[0]
    y = data.T[1]

    labels = np.sign(np.multiply(x, y))

    # shuffle the data
    perm = np.random.permutation(len(labels))
    data = data[perm]
    labels = labels[perm]
        
    return data, labels


def make_center_surround_data(
        examples_per_class: int,
        distance_from_origin: float
):
    """
    Create a 2-dimensional set of points, where half the points are drawn from
    one Gaussian centered on the origin and the other half form a ring around
    the first class

    PARAMETERS
    ----------
    examples_per_class      An integer determining how much data we'll generate

    distance_from_origin    All points from one of the Gaussians will have their
                            coordinates updated to have their distance from the
                            origin increased by this ammount. Should be
                            non-negative.

    RETURNS
    -------
    data      A numpy array of 2 columns (dimensions) and 2*examples_per_class rows

    labels    A numpy vector with 2*examples_per_class, with a +1 or -1 in each
              element. The jth element is the label of the jth example"""

    mean = [0, 0]
    cov = [[1, 0], [0, 1]]

    # negative class -1
    negData = np.random.multivariate_normal(mean, cov, examples_per_class)

    # positive class 1
    posData = np.random.multivariate_normal(mean, cov, examples_per_class)

    # now....treat the positive class as having been drawn from phase, magnitude
    # coordinates and manipulate the magnitude so the mean distance of the points
    # from the origin is 4...and make sure the distribution of phase is all the way
    # way around the circle
    magnitude = posData.T[0, :] + distance_from_origin
    phase = posData.T[1, :] * 2

    # now go back to cartesian coordinates
    x = magnitude * np.cos(phase)
    y = magnitude * np.sin(phase)

    # and stick it back in the array
    posData.T[0, :] = x
    posData.T[1, :] = y

    # wrap it up and return it.
    negL = np.ones(examples_per_class) * -1
    posL = np.ones(examples_per_class)
    data = np.concatenate([posData, negData])
    labels = np.concatenate([posL, negL])

    # shuffle the data
    perm = np.random.permutation(len(labels))
    data = data[perm]
    labels = labels[perm]
        
    return data, labels


def make_spiral_data(examples_per_class):
    """
    Create a 2-dimensional set of points in two interwoven spirals. All elements
    in a single spiral share a label (either +1 or -1, depending on the spiral)

    PARAMETERS
    ----------
    examples_per_class      An integer determining how much data we'll generate

    RETURNS
    -------
    data      A numpy array of 2 columns (dimensions) and 2*examples_per_class rows

    labels    A numpy vector with 2*examples_per_class, with a +1 or -1 in each
              element. The jth element is the label of the jth example"""

    theta = np.sqrt(np.random.rand(examples_per_class))*2*pi

    # make points in a spiral that have some randomness
    r_a = 2*theta + pi
    temp = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    negData = temp + np.random.randn(examples_per_class, 2)

    # make points in a spiral offset from the first one that have some randomness
    r_b = -2*theta - pi
    temp = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    posData = temp + np.random.randn(examples_per_class, 2)

    # give labels to the data
    negL = np.ones(examples_per_class) * -1
    posL = np.ones(examples_per_class)

    # return the data
    data = np.concatenate([posData, negData])
    labels = np.concatenate([posL, negL])

    # shuffle the data
    perm = np.random.permutation(len(labels))
    data = data[perm]
    labels = labels[perm]
        
    return data, labels
