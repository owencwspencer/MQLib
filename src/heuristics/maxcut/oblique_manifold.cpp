#include "heuristics/maxcut/oblique_manifold.h"

ObliqueManifold::ObliqueManifold(int p, int n)
    : p_(p) {
}

void ObliqueManifold::Proj(const std::vector<std::vector<double>>& X, 
                           std::vector<std::vector<double>>& H) const {
  int rows = X.size();
  int cols = X[0].size();

  std::vector<double> inners(rows, 0.0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      inners[i] += X[i][j] * H[i][j];
    }
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      H[i][j] = H[i][j] - X[i][j] * inners[i];
    }
  }
}

void ObliqueManifold::Normalize(std::vector<std::vector<double>>& X) const {
  int rows = X.size();
  int cols = X[0].size();

  std::vector<double> nrms(rows, 0.0);
  for (int i = 0; i < rows; i++) {
    double row_sum_squared = 0.0;
    for (int j = 0; j < cols; j++) {
      row_sum_squared += X[i][j] * X[i][j];
    }
    nrms[i] = std::sqrt(row_sum_squared);
  }

  for (int i = 0; i < rows; i++) {
    if (nrms[i] > 1e-10) {
      double inv_norm = 1.0 / nrms[i];
      for (int j = 0; j < cols; j++) {
        X[i][j] *= inv_norm;
      }
    } else {
      double value = 1.0 / std::sqrt(static_cast<double>(cols));
      for (int j = 0; j < cols; j++) {
        X[i][j] = value;
      }
    }
  }
}

void ObliqueManifold::Retr(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& U,
                          std::vector<std::vector<double>>& Y,
                          double t) const {
  int rows = X.size();
  int cols = X[0].size();

  Y.resize(rows, std::vector<double>(cols));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Y[i][j] = X[i][j] + t * U[i][j];
    }
  }
  
  Normalize(Y);
}

double ObliqueManifold::Inner(const std::vector<std::vector<double>>& U,
                              const std::vector<std::vector<double>>& V) const {
  int rows = U.size();
  int cols = U[0].size();

  double result = 0.0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result += U[i][j] * V[i][j];
    }
  }
  
  return result;
}

double ObliqueManifold::Norm(const std::vector<std::vector<double>>& X,
                            const std::vector<std::vector<double>>& U) const {
  int rows = U.size();
  int cols = U[0].size();
  
  double sum_squared = 0.0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      sum_squared += U[i][j] * U[i][j];
    }
  }
  
  return std::sqrt(sum_squared);
}

void ObliqueManifold::EGrad2RGrad(const std::vector<std::vector<double>>& X, 
                                  std::vector<std::vector<double>>& G) const {
  Proj(X, G);
}

double ObliqueManifold::TypicalDist() const {
  return std::sqrt(static_cast<double>(p_));
}