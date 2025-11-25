from emcee.autocorr import AutocorrError


def sample_until_convergence(p0,
                             sampler,
                             n_each,
                             max_iter,
                             verbose=False,
                             **kwargs):
    state = p0
    for i in range(max_iter):
        state = sampler.run_mcmc(state, n_each, progress=verbose, **kwargs)
        try:
            sampler.get_autocorr_time()
            return 0  # converged
        except AutocorrError as e:
            if verbose:
                print(f"Not converged yet: {i}")
                print(e)

            continue
    return 1  # not converged
