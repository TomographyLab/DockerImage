# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

import unittest
from tomolab._removed.ilang import verbose

verbose.set_verbose_low()


class TestSequenceMetropolisHastings(unittest.TestCase):
    """Sequence of tests for the Metropolis Hastings sampler"""

    def setUp(self):
        pass

    def test_sample(self):
        """.."""
        pass


class TestSequenceGradientDescent(unittest.TestCase):
    """Sequence of tests for the Gradient Descent sampler"""

    def setUp(self):
        pass

    def test_sample(self):
        """.."""
        pass


if __name__ == '__main__':
    unittest.main()
