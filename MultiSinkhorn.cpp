#include "MultiSinkhorn.h"

#include "MessagePassing.h"
#include "CongestionOperator.h"

#include <limits>


namespace sinkhorn
{
using namespace messages;

static double EPSILON = std::numeric_limits<double>::epsilon();

void MultimarginalSinkhorn::iterate(std::vector<Ref<MatrixXd>>& potentials) {
    size_t num_marginals = potentials.size();

    ArrayXXd conv;

    conv = contract(potentials, 0, kernel) + EPSILON;
    potentials[0] = rho_0.array() / conv;

    size_t k;
    for (k=1; k < num_marginals - 1; k++) {
        conv = contract(potentials, k, kernel) + EPSILON;
        potentials[k] = running->operator()(conv).array() / conv;  // KL-prox of the conv and divide
    }

    conv = contract(potentials, num_marginals-1, kernel) + EPSILON;
    potentials[num_marginals-1] = terminal->operator()(conv).array() / conv;

}

void MultimarginalSinkhorn::run(std::vector<Ref<MatrixXd>>& potentials, int num_iterations) {
    for (int i = 0; i < num_iterations; i++) {
        this->iterate(potentials);
    }
}

std::vector<MatrixXd> MultimarginalSinkhorn::get_marginals(std::vector<Ref<MatrixXd>>& potentials) {
    return compute_marginals(potentials, kernel);
}

} // namespace sinkhorn


