import numpy as np
import sys
from Models.check_argument import check_missingness, check_intmax, check_prop

class MCMCInfer:
    def __init__(self, model, **kwargs):
        """Adaptive Markov chain Monte Carlo simulation for SSMs using
            Robust Adaptive Metropolis algorithm by Vihola (2012). Several different
            MCMC sampling schemes are implemented, see parameter
            arguments, package vignette, Vihola, Helske, Franks (2020) and Helske and
            Vihola (2021) for details.
        Args:
            model: bssm model class object.
            iter (int): A positive integer defining the total number of MCMC iterations.
                  Suitable value depends on the model, data, and the choice of specific
                 algorithms mcmc_type and sampling_method. As increasing
                iter also increases run time, it is generally good idea to first
                 test the performance with a small values, e.g., less than 10000.
            output_type: 'full' - default, returns posterior samples from the posterior
                \eqn{p(\alpha, \theta | y)}); "theta" - for marginal posterior of
                theta);  "summary" -  return the mean and variance estimates of the
                 states and posterior samples of theta.
            burnin (int): A positive integer defining the length of the burn-in period
                    which is disregarded from the results. Defaults to iter/2.
                    Note that all MCMC algorithms of bssm use adaptive MCMC during the
                    burn-in period in order to find good proposal distribution.
            thin (int): A positive integer defining the thinning rate. All the MCMC
                    algorithms in bssm use the jump chain representation (see refs),
                    and the thinning is applied to these blocks. Defaults to 1.
                    For IS-corrected methods, larger value can also be
                    statistically more effective.
                    Note: With output_type = "summary",the thinning does not affect
                    the computations of the summary statistics in case of pseudo-marginal methods.
            gamma (float): Tuning parameter for the adaptation of RAM algorithm. Must be
                between 0 and 1.
            target_acceptance (float): Target acceptance rate for MCMC between 0 and 1.
                    Defaults to 0.234.
            S (np.array): Matrix defining the initial value for the lower triangular matrix
               of the RAM algorithm, so that the covariance matrix of the Gaussian proposal
               distribution is \eqn{SS^H}. Note that for some parameters
               (currently the standard deviation, dispersion, and autoregressive parameters
               of the BSM and AR(1) models) the sampling is done in unconstrained parameter
                space, i.e. internal_theta = log(theta) (and logit(rho) or AR coefficient).
            end_adaptive_phase (bool):  True - S is held fixed after the burnin period.
                    Default is False.
            threads (int): Number of threads for state simulation. Positive integer (default is 1).
                Note that parallel computing is only used in the post-correction phase of
                IS-MCMC and when sampling the states in case of (approximate) Gaussian models.
            seed (int): Seed for the C++ RNG - would delete when using TFP
            local_approx (bool): True (default) - Gaussian approximation needed for some of the
            methods is performed at each iteration. False - approximation is updated only
            once at the start of the MCMC using the initial model.
            max_iter (int): Maximum number of iterations used in Gaussian approximation,
                as a positive integer. Default is 100 (although typically only few iterations are needed).
           conv_tol (float): Positive tolerance parameter used in Gaussian approximation.
           particles (int): A positive integer defining the number of state samples per
                MCMC iteration for models other than linear-Gaussian models.
                Ignored if mcmc_type is "approx" or "ekf". Suitable
                values depend on the model, the data, mcmc_type and sampling_method.
                While large values provide more accurate estimates, the run time also
                increases with respect to the number of particles,
                so it is generally a good idea to test the run time
                first with a small number of particles, e.g., less than 100.
            mcmc_type: Type of MCMC algorithm should be used for models other than
                        linear-Gaussian models.
                "pm" - for pseudo-marginal MCMC,
                "da" - for delayed acceptance version of PMCMC ,
                "approx" - for approximate inference based on the Gaussian
                        approximation of the model,
                "ekf" - for approximate inference using extended Kalman filter
                        (for ssm_nlg),
                or one of the three importance sampling type weighting schemes:
                "is3" - for simple importance sampling (weight is computed for each
                         MCMC iteration independently),
                "is2" - for jump chain importance sampling type weighting (default)
                "is1": for importance sampling type weighting where the number of
                        particles used for weight computations is proportional
                        to the length of the jump chain block.
            sampling_method: Method for state sampling when for models other than linear-Gaussian models.
                "psi" -  \eqn{\psi}-APF is used (default).
                "spdk" - non-sequential importance sampling based on Gaussian approximation. If \code{"ekf"}, particle filter based on EKF-proposals are used
                (only for ssm_nlg models).
            iekf_iter (int): Non-negative integer. The default zero corresponds to
                normal EKF, whereas iekf_iter > 0 corresponds to iterated EKF
                with iekf_iter iterations. Used only for models of class ssm_nlg.
            L_c,L_f (int): For ssm_sde models, Positive integer values defining
                the discretization levels for first and second stages (defined as 2^L).
                For pseudo-marginal methods pm, maximum of these is used.
            verbose (bool): True - prints a progress bar to the console.
                    Set to alse if number of iterations is less than 50.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        def_mcmc = getattr(self, "model_type")
        def_mcmc()

    def linear_gaussian(self, model):
        """ For linear model and Guassian innovation.
        Args:
            model: bssm model object

        Returns:
            output_mcmc:  output after the mcmc

        """
        # check the list of arguments
        check_missingness(model)
        check_intmax(self.seed, postivive=False, max_val=sys.maxsize)
        check_intmax(self.threads)
        check_intmax(self.thin, max_val=100)
        check_intmax(self.iter, max_val=1e12)
        check_intmax(self.burnin, postivive=False, max_val=1e12)
        check_prop(self.target_acceptance)
        check_prop(self.gamma)
        if self.burnin > self.iter:
            raise ValueError("Argument 'burnin' should be smaller than 'iter'.")
        if not self.end_adaptive_phase in [True, False]:
            raise TypeError("Argument 'end_adaptive_phase' should be TRUE or FALSE.")
        if not self.verbose in [True, False]:
            raise TypeError("Argument 'verbose' should be TRUE or FALSE.")
        if not hasattr(model, "theta"):
            raise AttributeError("No unknown parameters ('theta' has length of zero).")
        if model.model_name == "bsm_lg":
            pass
        if not hasattr(self, "S"):
            self.S = np.diag(0.1* np.maximum(0.1, np.abs(model.theta)))

    def _toR(self):
        """Convert to R object
        Returns:

        """




