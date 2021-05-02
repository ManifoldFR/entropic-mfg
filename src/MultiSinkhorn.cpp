#include <limits>

#include "MultiSinkhorn.h"

#include "MessagePassing.h"
#include "progressbar.h"
#include "metrics.h"



namespace sinkhorn
{

void MultimarginalSinkhorn::iterate() {

    size_t n_marg = potentials_.size();
    for (size_t t = 0; t < n_marg; t++) {
        _old_potentials[t] = potentials_[t];
    }

    ArrayXXd conv_pre_;

    size_t k = 0;
    utils::ProgressBar pbar{n_marg};

    conv_pre_ = messages::contract(potentials_, 0, kernel).array();
    potentials_[0] = rho_0.array() / (conv_pre_ + EPSILON);
    pbar.print_progress();

    for (k = 1; k < n_marg - 1; k++) {
        conv_pre_ = messages::contract(potentials_, k, kernel).array();
        // KL-prox of the conv_pre_ and divide
        potentials_[k] = (*running)(conv_pre_).array() / (conv_pre_ + EPSILON);
        pbar.print_progress();
    }

    conv_pre_ = messages::contract(potentials_, n_marg-1, kernel).array();
    potentials_[n_marg-1] = (*terminal)(conv_pre_).array() / (conv_pre_ + EPSILON);
    pbar.print_progress();

    double dist = hilbert_metric<double>(_old_potentials, potentials_);
    metric_vals_.push_back(dist);
}

bool MultimarginalSinkhorn::solve(size_t num_iterations, std::vector<MatrixXd> &potentials, bool verbose)
{
    verbose_ = verbose;
    if (verbose_)
        std::cout << BLUE << "> MultiSinkhorn solver" << RESET << std::endl;

    nsteps_ = potentials.size();
    metric_vals_.resize(0);
    _old_potentials.resize(nsteps_);
    potentials_.resize(nsteps_);

    for (size_t t = 0; t < nsteps_; t++) {
        potentials_[t].noalias() = potentials[t];
    }

    for (size_t i = 0; i < num_iterations; i++) {
        iterate();
        if (metric_vals_[i] < threshold_)
        {
            std::cout << "> Early stop." << std::endl;
            break;
        }
    }

    return true;
}

std::vector<MatrixXd> MultimarginalSinkhorn::get_marginals() {
    std::vector<MatrixXd> res(nsteps_);
    std::vector<ArrayXXd> marg_ = messages::compute_marginals(potentials_, kernel);
    for (size_t k = 0; k < nsteps_; k++) {
        res[k] = marg_[k].matrix();
    }
    return res;
}

} // namespace sinkhorn


