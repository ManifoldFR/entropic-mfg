/// @file Determine when to stop the itertions using
/// a Hilbert metric.
#include <Eigen/Core>

#include <vector>
#include <limits>

namespace sinkhorn
{

constexpr double EPSILON = std::numeric_limits<double>::epsilon();

using namespace Eigen;

template<typename S = double>
S hilbert_metric(const Matrix<S,-1,-1>& u, const Matrix<S,-1,-1>& v)
{
    auto a = u.array() + EPSILON;
    auto b = v.array() + EPSILON;
    auto diff = a.log() - b.log();
    S dmax = diff.maxCoeff();
    S dmin = diff.minCoeff();
    return dmax - dmin;
}

template <typename S = double>
S hilbert_metric(const std::vector<Matrix<S, -1, -1>> us,
                 const std::vector<Matrix<S, -1, -1>> vs)
{
    assert(us.size() == vs.size() && "us and vs should have the same lengths.");
    S result = 0.;
    size_t n = us.size();

    for (size_t t = 0; t < n; t++) {
        result += hilbert_metric(us[t], vs[t]);
    }
    return result;
}
}
