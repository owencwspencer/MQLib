#ifndef HEURISTICS_MAXCUT_MAX_CUT_PROBLEM_H_
#define HEURISTICS_MAXCUT_MAX_CUT_PROBLEM_H_

#include <vector>
#include "heuristics/maxcut/oblique_manifold.h"
#include "problem/max_cut_instance.h"

struct Store {
  std::vector<std::vector<double>> AY;
  std::vector<double> diagAYYt;
  bool has_AY = false;
};

class MaxCutProblem {
 public:
  MaxCutProblem(const MaxCutInstance& instance, ObliqueManifold& manifold, int p, int n);
  
  const MaxCutInstance& A;
  ObliqueManifold& M;

  double cost(const MaxCutProblem& problem, const std::vector<std::vector<double>>& Y, Store& store);
                    
  void grad(const MaxCutProblem& problem, Store& store, const std::vector<std::vector<double>>& Y,
            std::vector<std::vector<double>>& G);
                  
  void hess(const MaxCutProblem& problem, Store& store, const std::vector<std::vector<double>>& Y,
            const std::vector<std::vector<double>>& Ydot, std::vector<std::vector<double>>& H);
                  
  Store prepare(const MaxCutProblem& problem, const std::vector<std::vector<double>>& Y, const Store& store);

 private:
  std::vector<std::vector<double>> computeAY(const MaxCutInstance& mi, const std::vector<std::vector<double>>& Y, int n, int p);
                                        
  std::vector<double> computeDiagAYYt(const std::vector<std::vector<double>>& AY,
                                      const std::vector<std::vector<double>>& Y, int n, int p);
};

#endif