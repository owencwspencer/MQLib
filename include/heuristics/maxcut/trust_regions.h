#ifndef HEURISTICS_MAXCUT_TRUST_REGIONS_H_
#define HEURISTICS_MAXCUT_TRUST_REGIONS_H_

#include <vector>
#include <functional>
#include "heuristics/maxcut/max_cut_problem.h"

struct TrustRegionOptions {
  int verbosity = 2;
  int maxiter = 1000;
  int miniter = 0;
  double Delta_bar = 0.0;
  double Delta0 = 0.0;
  int maxinner = 500;
  double tolgradnorm = 1e-6;
  double tolcost = -1.0;
  double maxtime = -1.0;
  double rho_prime = 0.1;
  double rho_regularization = 1e3;
};

struct TRSOutput {
  std::vector<std::vector<double>> eta;
  std::vector<std::vector<double>> Heta;
  bool limitedbyTR;
};

double cost(const MaxCutProblem& problem, const std::vector<std::vector<double>>& Y, Store& store);
          
void grad(const MaxCutProblem& problem, Store& store, const std::vector<std::vector<double>>& Y,
          std::vector<std::vector<double>>& G);
        
void hess(const MaxCutProblem& problem, Store& store, const std::vector<std::vector<double>>& Y,
          const std::vector<std::vector<double>>& Ydot, std::vector<std::vector<double>>& H);
        
Store prepare(const MaxCutProblem& problem, const std::vector<std::vector<double>>& Y, const Store& store);

std::vector<std::vector<double>> TrustRegions(MaxCutProblem& problem, const std::vector<std::vector<double>>& Y0,
  TrustRegionOptions& opts);

TRSOutput TRS_tCG(const MaxCutProblem& problem, const std::vector<std::vector<double>>& x,
  const std::vector<std::vector<double>>& grad_x, double Delta, bool accept, const TrustRegionOptions& opts,
  Store& store);

bool StoppingCriterion(const MaxCutProblem& problem, const std::vector<std::vector<double>>& x,
	const std::vector<std::vector<double>>& grad_x, double cost, double grad_norm, double elapsed_time,
	const TrustRegionOptions& opts, int iter);

double ComputeModelValue(const MaxCutProblem& problem, const std::vector<std::vector<double>>& x,
	const std::vector<std::vector<double>>& eta, const std::vector<std::vector<double>>& Heta,
	const std::vector<std::vector<double>>& grad);

#endif