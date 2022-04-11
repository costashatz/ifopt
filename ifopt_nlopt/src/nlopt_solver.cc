/******************************************************************************
Copyright (c) 2022, Konstantinos Chatzilygeroudis. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of ETH ZURICH nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <nlopt.hpp>

#include <ifopt/nlopt_solver.h>

#include <iostream>

namespace ifopt {

static double nlopt_func(unsigned n, const double* x, double* gradient, void* my_func_data)
{
    double v = 0.;
    // for (unsigned i = 0; i < n; i++)
    //     v =+ x[i] * x[i];
    Problem* nlp = reinterpret_cast<Problem*>(my_func_data);
    if (gradient) { // if the algorithm is taking gradient information
        Eigen::VectorXd g = nlp->EvaluateCostFunctionGradient(x);

        // for (unsigned i = 0; i < n; i++)
        //     g[i] += 2. * x[i];

        Eigen::VectorXd::Map(gradient, n) = g;

        // nlp->SaveCurrent();
        // std::cout << nlp->GetVariableValues().transpose() << " vs " << g.transpose() << " --> " << nlp->EvaluateCostFunction(x) << std::endl;
    }

    v += nlp->EvaluateCostFunction(x);
    return v;
}

static void nlopt_constraint(unsigned m, double* result, unsigned n, const double* x, double* gradient, void* my_func_data)
{
    // static int iter = 0;
    Problem* nlp = reinterpret_cast<Problem*>(my_func_data);
    // std::cout << m << " vs " << nlp->GetNumberOfConstraints() << std::endl;
    auto bounds = nlp->GetBoundsOnConstraints();

    Eigen::VectorXd r = nlp->EvaluateConstraints(x);
    Eigen::VectorXd res(2 * r.size());
    for (int i = 0; i < r.size(); i++) {
        res[i] = r[i] - bounds[i].upper_;
        res[i + r.size()] = bounds[i].lower_ - r[i];
    }
    // std::cout << r.transpose() << std::endl;
    // std::cout << res.transpose() << std::endl;


    // std::cout << iter++ << std::endl;

    Eigen::VectorXd::Map(result, m) = res;

    if (gradient) {
        Problem::Jacobian jac = nlp->GetJacobianOfConstraints();
        assert(jac.rows() == r.size() && jac.cols() == n);

        // std::cout << jac.rows() << "x" << jac.cols() << std::endl;
        // std::cout << "G:" << jac << std::endl;
        // std::cout << std::endl;
        // std::cout << "G2: " << std::endl;
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < r.size(); j++) {
        //         std::cout << jac.coeff(i, j) << " " ;
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;

        // std::cout << "G3:" << std::endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double g = jac.coeff(i % r.size(), j);
                if (i >= r.size())
                    g = -g;
                // std::cout << g << " ";
                gradient[i * n + j] = g;
            }
            // std::cout << std::endl;
        }
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < r.size(); j++) {
        //         gradient[i*m + j] = jac.coeff(i, j);
        //         gradient[i*m + j + r.size()] = jac.coeff(i, j);
        //     }
        // }
    }
    // else {
    //     std::cout << "No gradient" << std::endl;
    // }
    // std::cin.get();
}

NLoptSolver::NLoptSolver() {}

void
NLoptSolver::Solve (Problem& nlp)
{
    nlopt::opt opt = nlopt::opt(nlopt::algorithm::LD_SLSQP, nlp.GetNumberOfOptimizationVariables());

    Eigen::VectorXd x_init = nlp.GetOptVariables()->GetValues();

    opt.set_min_objective(nlopt_func, &nlp);

    opt.add_inequality_mconstraint(nlopt_constraint, &nlp, std::vector<double>(nlp.GetNumberOfConstraints() * 2, 1e-2));

    std::vector<double> lower, upper;
    auto bounds = nlp.GetBoundsOnOptimizationVariables();
    for (size_t i = 0; i < bounds.size(); i++) {
        lower.push_back(bounds[i].lower_);
        upper.push_back(bounds[i].upper_);

        if (x_init[i] < lower.back())
            x_init[i] = lower.back();
        else if (x_init[i] > upper.back())
            x_init[i] = upper.back();
    }

    nlp.SetVariables(x_init.data());

    opt.set_lower_bounds(lower);
    opt.set_upper_bounds(upper);
    // opt.set_xtol_rel(1e-5);
    // opt.set_maxeval(20000);

    std::vector<double> x(x_init.size());
    Eigen::VectorXd::Map(&x[0], x_init.size()) = x_init;

    double min_val;
    try {
        opt.optimize(x, min_val);
    }
    catch (nlopt::roundoff_limited& e) {
        // In theory it's ok to ignore this error
        // std::cerr << "[NLOptNoGrad]: " << e.what() << std::endl;
    }
    catch (std::invalid_argument& e) {
        // In theory it's ok to ignore this error
        // std::cerr << "[NLOptNoGrad]: " << e.what() << std::endl;
    }
    catch (std::runtime_error& e) {
        // In theory it's ok to ignore this error
        // std::cerr << "[NLOptGrad]: " << e.what() << std::endl;
    }
}

double
NLoptSolver::GetTotalWallclockTime()
{
  return 0.;
}

} /* namespace ifopt */
