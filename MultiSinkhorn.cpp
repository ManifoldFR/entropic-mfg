#include "MultiSinkhorn.h"

#include "MessagePassing.h"
#include "CongestionOperator.h"


namespace sinkhorn
{
using namespace messages;

void MultimarginalSinkhorn::iterate(std::vector<MatrixXd>& potentials) {
    size_t num_marginals = potentials.size();

    ArrayXXd conv;


    conv = contract(potentials, 0, kernel);
    potentials[0] = rho_0.array() / conv;

    for (size_t k=1; k < num_marginals - 1; k++) {
        conv = contract(potentials, k, kernel);
        potentials[k] = running->operator()(potentials[k]).array() / conv;
    }

    conv = contract(potentials, num_marginals-1, kernel);
    potentials[num_marginals-1] = terminal->operator()(potentials[num_marginals-1]).array() / conv;

}

} // namespace sinkhorn


