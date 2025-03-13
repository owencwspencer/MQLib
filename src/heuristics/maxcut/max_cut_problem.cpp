#include "heuristics/maxcut/max_cut_problem.h"

MaxCutProblem::MaxCutProblem(const MaxCutInstance& instance, ObliqueManifold& manifold, int p, int n)
    : A(instance), M(manifold) {
}

std::vector<std::vector<double>> computeAY(const MaxCutInstance& mi, const std::vector<std::vector<double>>& Y, int n, int p) {
  std::vector<std::vector<double>> AY(n, std::vector<double>(p, 0.0));

  for (auto iter = mi.get_all_edges_begin(); iter != mi.get_all_edges_end(); ++iter) {
    int i = iter->first.first;
    int j = iter->first.second;
    double w_ij = iter->second;

    for (int k = 0; k < p; k++) {
      AY[i][k] += w_ij * Y[j][k];
      AY[j][k] += w_ij * Y[i][k];
    }
  }

  return AY;
}

std::vector<double> computeDiagAYYt(const std::vector<std::vector<double>>& AY,
                                    const std::vector<std::vector<double>>& Y, int n, int p) {
  std::vector<double> diagAYYt(n, 0.0);
  
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
          diagAYYt[i] += AY[i][j] * Y[i][j];
      }
  }
  
  return diagAYYt;
}

Store prepare(const MaxCutProblem& problem, const std::vector<std::vector<double>>& Y, const Store& store) {
  Store new_store = store;

  if (!new_store.has_AY) {
    int n = Y.size();
    int p = Y[0].size();
    
    new_store.AY = computeAY(problem.A, Y, n, p);
    new_store.diagAYYt = computeDiagAYYt(new_store.AY, Y, n, p);
    
    new_store.has_AY = true;
  }

  return new_store;
}

double cost(const MaxCutProblem& problem, const std::vector<std::vector<double>>& Y, Store& store) {
  store = prepare(problem, Y, store);

  double cost_value = 0.0;
  for (size_t i = 0; i < store.diagAYYt.size(); i++) {
    cost_value += store.diagAYYt[i];
  }
  
  return 0.5 * cost_value;
}

void grad(const MaxCutProblem& problem, Store& store, const std::vector<std::vector<double>>& Y,
  std::vector<std::vector<double>>& G) {
  store = prepare(problem, Y, store);

  int n = Y.size();
  int p = Y[0].size();
  
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
          G[i][j] = store.AY[i][j] - Y[i][j] * store.diagAYYt[i];
      }
  }

  problem.M.Proj(Y, G);
}

void hess(const MaxCutProblem& problem, Store& store, const std::vector<std::vector<double>>& Y,
  const std::vector<std::vector<double>>& Ydot, std::vector<std::vector<double>>& H) {
  store = prepare(problem, Y, store);
  
  int n = Y.size();
  int p = Y[0].size();
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      H[i][j] = 0.0;
    }
  }
  
  for (auto iter = problem.A.get_all_edges_begin(); iter != problem.A.get_all_edges_end(); ++iter) {
    int i = iter->first.first;
    int j = iter->first.second;
    double w = iter->second;
    
    for (int k = 0; k < p; k++) {
      H[i][k] += w * Ydot[j][k];
      H[j][k] += w * Ydot[i][k];
    }
  }
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      H[i][j] -= Ydot[i][j] * store.diagAYYt[i];
    }
  }
  
  problem.M.Proj(Y, H);
}