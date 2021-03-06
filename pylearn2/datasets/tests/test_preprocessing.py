from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import GlobalContrastNormalization
from pylearn2.utils import as_floatX
import numpy as np

class GlobalContrastNormalizationTests:
    """Tests for the GlobalContrastNormalization class """

    def test_zero_vector(self):
        """ Test that passing in the zero vector does not result in
            a divide by 0 """

        dataset      = DenseDesignMatrix(X = as_floatX(np.zeros(())))

        #the settings of subtract_mean and use_norm are not relevant to
        #the test
        #std_bias = 0.0 is the only value for which there should be a risk
        #of failure occurring
        preprocessor = GlobalContrastNormalization( subtract_mean = True,
                                                    std_bias = 0.0,
                                                    use_norm = False)

        dataset.apply(preprocessor)

        result = dataset.get_design_matrix()

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_unit_norm(self):
        """ Test that using std_bias = 0.0 and use_norm = True
            results in vectors having unit norm """

        tol = 1e-5

        num_examples = 5
        num_features = 10

        rng = np.random.RandomState([1,2,3])

        X = as_floatX(rng.randn(5,10))

        dataset = DenseDesignMatrix( X = X )

        #the setting of subtract_mean is not relevant to the test
        #the test only applies when std_bias = 0.0 and use_norm = True
        preprocessor = GlobalContrastNormalization( subtract_mean = False,
                                                    std_bias = 0.0,
                                                    use_norm = True)

        dataset.apply(preprocessor)

        result = dataset.get_design_matrix()

        norms = np.sqrt(np.square(result).sum(axis=1))

        max_norm_error = np.abs(norms-1.).max()

        tol = 3e-5

        assert max_norm_error < tol
