#include <math.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include "heuristics/maxcut/boumal2016.h"
#include "util/random.h"

MaxCutSolution Boumal2016::ExtractCutFromY(const MaxCutInstance& mi, const std::vector<std::vector<double>>& Y) {
  int n = Y.size();
  int p = Y[0].size();
  
  double best_cut_value = -std::numeric_limits<double>::infinity();
  std::vector<int> best_assignments(n);
  
  for (int col = 0; col < p; col++) {
    std::vector<int> assignments(n);
    for (int i = 0; i < n; i++) {
      assignments[i] = (Y[i][col] > 0) ? 1 : -1;
    }
    
    MaxCutSolution temp_sol(assignments, mi, this);
    double cut_value = temp_sol.get_weight();
    
    if (cut_value > best_cut_value) {
      best_cut_value = cut_value;
      best_assignments = assignments;
    }
  }
  
  return MaxCutSolution(best_assignments, mi, this);
}

void Boumal2016::GenerateY0(const ObliqueManifold& manifold, int n, int p, std::vector<std::vector<double>>& Y0) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      Y0[i][j] = Random::RandDouble() * 2.0 - 1.0;
    }
  }
  manifold.Normalize(Y0);
}

void Boumal2016::PerturbSolution(std::vector<std::vector<double>>& Y, const MaxCutSolution& sol,
                                 const MaxCutProblem& problem, double perturbation_factor) {
  int n = Y.size();
  int p = Y[0].size();
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      double base_value = (sol.get_assignments()[i] > 0) ? 1.0 : -1.0;
      Y[i][j] = base_value + perturbation_factor * (Random::RandDouble() * 2.0 - 1.0);
    }
  }
  
  problem.M.Normalize(Y);
}

bool Boumal2016::OptimizationLoop(const MaxCutInstance& mi, MaxCutProblem& problem, std::vector<std::vector<double>>& Y,
                                  TrustRegionOptions& opts, int max_non_improving, double& best_cut_value) {
  int non_improving_count = 0;
  bool continue_optimization = true;
  
  while (continue_optimization && non_improving_count < max_non_improving) {
    Y = TrustRegions(problem, Y, opts);
    
    MaxCutSolution sol = ExtractCutFromY(mi, Y);
    sol.AllBest1Swap();
    sol.AllBest2Swap();
    
    if (sol.get_weight() > best_cut_value + 1e-6) {
      best_cut_value = sol.get_weight();
      non_improving_count = 0;
    } else {
      non_improving_count++;
    }
    
    if (!Report(sol)) {
      return false;
    }
    
    double perturbation = 0.1 + 0.05 * non_improving_count;
    PerturbSolution(Y, sol, problem, perturbation);
  }
  
  return true;
}

Boumal2016::Boumal2016(const MaxCutInstance& mi, double runtime_limit, bool validation, MaxCutCallback* mc) :
  MaxCutHeuristic(mi, runtime_limit, validation, mc) {
  
  int n = mi.get_size();
  int p = std::ceil(std::sqrt(8.0*n+1.0)/2.0);
  ObliqueManifold manifold(p, n);
  MaxCutProblem problem(mi, manifold, p, n);
  
  const int max_iter_per_run = 500;
  const int max_non_improving = 10;
  
  TrustRegionOptions opts;
  opts.verbosity = 2;  
  opts.maxinner = 500;
  opts.Delta_bar = problem.M.TypicalDist(); 
  opts.Delta0 = opts.Delta_bar / 8.0; 
  opts.maxiter = max_iter_per_run;
  
  std::vector<std::vector<double>> Y0(n, std::vector<double>(p));
  GenerateY0(manifold, n, p, Y0);
  std::vector<std::vector<double>> Y = Y0;
  
  double best_cut_value = -std::numeric_limits<double>::infinity();
  bool continue_optimization = true;
  
  continue_optimization = OptimizationLoop(mi, problem, Y, opts, max_non_improving, best_cut_value);
  
  for (int restart = 0; restart < 20 && continue_optimization; restart++) {
    GenerateY0(manifold, n, p, Y0);
    Y = Y0;
    
    continue_optimization = OptimizationLoop(mi, problem, Y, opts, max_non_improving, best_cut_value);
  }
}