#include "heuristics/maxcut/trust_regions.h"
#include "heuristics/maxcut/max_cut_problem.h"
#include "heuristics/maxcut/max_cut_solution.h"
#include "heuristics/maxcut/boumal2016.h"
#include "heuristics/maxcut/oblique_manifold.h"
#include <cmath>
#include <sstream>
#include <algorithm>
#include <chrono>

bool StoppingCriterion(const MaxCutProblem& problem, const std::vector<std::vector<double>>& x,
  const std::vector<std::vector<double>>& grad_x, double cost, double grad_norm, double elapsed_time,
  const TrustRegionOptions& opts, int iter) {

  if (iter < opts.miniter) {
    return false;
  }

  if (cost <= opts.tolcost ||
    grad_norm < opts.tolgradnorm ||
    elapsed_time >= opts.maxtime ||
    iter >= opts.maxiter) {
    return true;
  }

  return false;
}

double ComputeModelValue(const MaxCutProblem& problem, const std::vector<std::vector<double>>& x,
												 const std::vector<std::vector<double>>& eta, const std::vector<std::vector<double>>& Heta,
												 const std::vector<std::vector<double>>& grad) {
  return problem.M.Inner(eta, grad) + 0.5 * problem.M.Inner(eta, Heta);
}

TRSOutput TRS_tCG(const MaxCutProblem& problem, const std::vector<std::vector<double>>& x,
          const std::vector<std::vector<double>>& grad_x, double Delta, bool accept, const TrustRegionOptions& opts,
          Store& store) {
  int n = x.size();
  int p = x[0].size();

  TRSOutput result;
  result.limitedbyTR = false;

  double kappa = 0.1;
  double theta = 1.0;

  std::vector<std::vector<double>> eta(n, std::vector<double>(p, 0.0));
  std::vector<std::vector<double>> Heta(n, std::vector<double>(p, 0.0));

  std::vector<std::vector<double>> r(n, std::vector<double>(p));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      r[i][j] = grad_x[i][j];
    }
  }

  double r_r = problem.M.Inner(r, r);
  double norm_r = std::sqrt(r_r);
  double norm_g = norm_r;

  std::vector<std::vector<double>> z = r;

  double z_r = problem.M.Inner(z, r);

  std::vector<std::vector<double>> mdelta = z;
  
  double d_Pd = z_r;
  double e_Pd = 0.0;
  double e_Pe = 0.0;

  double model_value = ComputeModelValue(problem, x, eta, Heta, grad_x);

  for (int j = 0; j < opts.maxinner; j++) {
    std::vector<std::vector<double>> Hmdelta(n, std::vector<double>(p));
    hess(problem, store, x, mdelta, Hmdelta);

    double d_Hd = problem.M.Inner(mdelta, Hmdelta);

    double alpha = z_r / d_Hd;

    double e_Pe_new = e_Pe + 2.0 * alpha * e_Pd + alpha * alpha * d_Pd;

    if (d_Hd <= 0 || e_Pe_new >= Delta * Delta) {
      double tau = (-e_Pd + std::sqrt(e_Pd * e_Pd + d_Pd * (Delta * Delta - e_Pe))) / d_Pd;
      
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
          eta[i][j] = eta[i][j] - tau * mdelta[i][j];
          Heta[i][j] = Heta[i][j] - tau * Hmdelta[i][j];
        }
      }
      result.limitedbyTR = true;
      break;
    }

    e_Pe = e_Pe_new;

    std::vector<std::vector<double>> new_eta(n, std::vector<double>(p));
    std::vector<std::vector<double>> new_Heta(n, std::vector<double>(p));
    
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
        new_eta[i][j] = eta[i][j] - alpha * mdelta[i][j];
        new_Heta[i][j] = Heta[i][j] - alpha * Hmdelta[i][j];
      }
    }

    double new_model_value = ComputeModelValue(problem, x, new_eta, new_Heta, grad_x);
    
    if (new_model_value >= model_value) {
      break;
    }
    
    eta = new_eta;
    Heta = new_Heta;
    model_value = new_model_value;
    
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
        r[i][j] = r[i][j] - alpha * Hmdelta[i][j];
      }
    }

    r_r = problem.M.Inner(r, r);
    norm_r = std::sqrt(r_r);
    
    if (j >= opts.miniter && norm_r <= norm_g * std::min(std::pow(norm_g, theta), kappa)) {
      break;
    }

    z = r;
    
    double zold_rold = z_r;
    
    z_r = problem.M.Inner(z, r);
    
    double beta = z_r / zold_rold;
    
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
        mdelta[i][j] = z[i][j] + beta * mdelta[i][j];
      }
    }
    
    problem.M.Proj(x, mdelta);
    
    e_Pd = beta * (e_Pd + alpha * d_Pd);
    
    d_Pd = z_r + beta * beta * d_Pd;
  }

  result.eta = eta;
  result.Heta = Heta;
  
  return result;
}

std::vector<std::vector<double>> TrustRegions(MaxCutProblem& problem, const std::vector<std::vector<double>>& Y0,
                        TrustRegionOptions& opts) {
  int n = problem.A.get_size();
  int p = Y0.size() > 0 ? Y0[0].size() : 0;
  
  std::vector<std::vector<double>> x = Y0;
  
  if (opts.Delta_bar <= 0.0) {
    opts.Delta_bar = problem.M.TypicalDist();
  }
  if (opts.Delta0 <= 0.0) {
    opts.Delta0 = opts.Delta_bar / 8.0;
  }
  
  int k = 0;
  bool accept = true;
  
  auto start_time = std::chrono::steady_clock::now();
  double elapsed_time = 0.0;
  
  Store store;
  
  double fx = cost(problem, x, store);
  std::vector<std::vector<double>> fgradx(n, std::vector<double>(p));
  grad(problem, store, x, fgradx);
  
  double norm_grad = problem.M.Norm(x, fgradx);
  double Delta = opts.Delta0;
  
  while (true) {
    elapsed_time = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start_time).count();
    
    if (StoppingCriterion(problem, x, fgradx, fx, norm_grad, elapsed_time, opts, k)) {
      break;
    }
    
    TRSOutput trsoutput = TRS_tCG(problem, x, fgradx, Delta, accept, opts, store);
    
    std::vector<std::vector<double>> eta = trsoutput.eta;
    std::vector<std::vector<double>> Heta = trsoutput.Heta;
    bool limitedbyTR = trsoutput.limitedbyTR;
    
    std::vector<std::vector<double>> x_prop(n, std::vector<double>(p));
    problem.M.Retr(x, eta, x_prop);
    
    Store temp_store;
    double fx_prop = cost(problem, x_prop, temp_store);
    
    double rhonum = fx - fx_prop;
    std::vector<std::vector<double>> vecrho(n, std::vector<double>(p));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
        vecrho[i][j] = fgradx[i][j] + 0.5 * Heta[i][j];
      }
    }
    
    double rhoden = -problem.M.Inner(eta, vecrho);
    
    double rho_reg_numeric = std::max(1.0, std::abs(fx)) * std::numeric_limits<double>::epsilon() * opts.rho_regularization;
    rhonum = rhonum + rho_reg_numeric;
    rhoden = rhoden + rho_reg_numeric;
    double rho = rhonum / rhoden;
    
    bool model_decreased = (rhoden >= 0);
    
    accept = false;
    bool should_accept = (model_decreased && rho > opts.rho_prime);
    
    if (should_accept) {
      accept = true;
      x = x_prop;
      fx = fx_prop;
      store = temp_store;
      grad(problem, store, x, fgradx);
      norm_grad = problem.M.Norm(x, fgradx);
    }
    
    if (!accept || rho < 0.25 || std::isnan(rho)) {
      Delta = Delta / 4.0;
    } else if (rho > 0.75 && limitedbyTR) {
      Delta = std::min(2.0 * Delta, opts.Delta_bar);
    }
    
    k++;
  }
  
  return x;
}