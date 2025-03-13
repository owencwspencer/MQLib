#ifndef HEURISTICS_MAXCUT_OBLIQUE_MANIFOLD_H_
#define HEURISTICS_MAXCUT_OBLIQUE_MANIFOLD_H_

#include <vector>
#include <cmath>

class ObliqueManifold {
public:
  ObliqueManifold(int p, int n);
  
  void Proj(const std::vector<std::vector<double>>& X, 
            std::vector<std::vector<double>>& H) const;
            
  void Normalize(std::vector<std::vector<double>>& X) const;

  void Retr(const std::vector<std::vector<double>>& X,
            const std::vector<std::vector<double>>& U,
            std::vector<std::vector<double>>& Y,
            double t = 1.0) const;

  double Inner(const std::vector<std::vector<double>>& U,
               const std::vector<std::vector<double>>& V) const;
  
  double Norm(const std::vector<std::vector<double>>& X,
              const std::vector<std::vector<double>>& U) const;

  void EGrad2RGrad(const std::vector<std::vector<double>>& X, 
                   std::vector<std::vector<double>>& G) const;

  double TypicalDist() const;

private:
  int p_;
};

#endif