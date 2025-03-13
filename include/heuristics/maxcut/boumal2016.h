#ifndef HEURISTICS_MAXCUT_BOUMAL_2016_H_
#define HEURISTICS_MAXCUT_BOUMAL_2016_H_

#include <vector>
#include "heuristics/maxcut/max_cut_problem.h"
#include "heuristics/maxcut/max_cut_solution.h"
#include "heuristics/maxcut/oblique_manifold.h"
#include "heuristics/maxcut/trust_regions.h"
#include "problem/max_cut_heuristic.h"

class Boumal2016 : public MaxCutHeuristic {
 public:
  Boumal2016(const MaxCutInstance& mi, double runtime_limit, bool validation, MaxCutCallback* mc);

  MaxCutSolution ExtractCutFromY(const MaxCutInstance& mi, const std::vector<std::vector<double>>& Y);

 private:
  void GenerateY0(const ObliqueManifold& manifold, int n, int p, std::vector<std::vector<double>>& Y0);

  void PerturbSolution(std::vector<std::vector<double>>& Y, const MaxCutSolution& sol,
                       const MaxCutProblem& problem, double perturbation_factor);

  bool OptimizationLoop(const MaxCutInstance& mi, MaxCutProblem& problem, std::vector<std::vector<double>>& Y,
                        TrustRegionOptions& opts, int max_non_improving, double& best_cut_value);
};

#endif