#ifndef CRM_BVP_JACOBIAN_HPP
#define CRM_BVP_JACOBIAN_HPP

#include "CRM.hpp"
#include "CRMDYN.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace CRMCatheterModel {

// Scalar dual number for forward-mode AD
// Each dual holds a value and a single directional derivative
struct Dual {
    double val;    // Primal value
    double deriv;  // Derivative in the seeded direction

    Dual() : val(0.0), deriv(0.0) {}
    Dual(double v) : val(v), deriv(0.0) {}
    Dual(double v, double d) : val(v), deriv(d) {}

    // Arithmetic operators
    Dual operator+(const Dual& other) const {
        return Dual(val + other.val, deriv + other.deriv);
    }

    Dual operator-(const Dual& other) const {
        return Dual(val - other.val, deriv - other.deriv);
    }

    Dual operator*(const Dual& other) const {
        return Dual(val * other.val, val * other.deriv + deriv * other.val);
    }

    Dual operator/(const Dual& other) const {
        double inv = 1.0 / other.val;
        return Dual(val * inv, (deriv * other.val - val * other.deriv) * inv * inv);
    }

    Dual operator+(double scalar) const {
        return Dual(val + scalar, deriv);
    }

    Dual operator-(double scalar) const {
        return Dual(val - scalar, deriv);
    }

    Dual operator*(double scalar) const {
        return Dual(val * scalar, deriv * scalar);
    }

    Dual operator/(double scalar) const {
        return Dual(val / scalar, deriv / scalar);
    }

    Dual operator-() const {
        return Dual(-val, -deriv);
    }

    // Compound assignment operators
    Dual& operator+=(const Dual& other) {
        val += other.val;
        deriv += other.deriv;
        return *this;
    }

    Dual& operator-=(const Dual& other) {
        val -= other.val;
        deriv -= other.deriv;
        return *this;
    }

    Dual& operator*=(const Dual& other) {
        double new_deriv = val * other.deriv + deriv * other.val;
        val *= other.val;
        deriv = new_deriv;
        return *this;
    }

    Dual& operator*=(double scalar) {
        val *= scalar;
        deriv *= scalar;
        return *this;
    }
};

// Math functions for Dual
inline Dual sqrt(const Dual& x) {
    double sqrt_val = std::sqrt(x.val);
    return Dual(sqrt_val, x.deriv / (2.0 * sqrt_val));
}

inline Dual sin(const Dual& x) {
    return Dual(std::sin(x.val), x.deriv * std::cos(x.val));
}

inline Dual cos(const Dual& x) {
    return Dual(std::cos(x.val), -x.deriv * std::sin(x.val));
}

inline Dual exp(const Dual& x) {
    double exp_val = std::exp(x.val);
    return Dual(exp_val, x.deriv * exp_val);
}

inline Dual fabs(const Dual& x) {
    return Dual(std::fabs(x.val), x.val >= 0 ? x.deriv : -x.deriv);
}

// Scalar-dual mixed operations
inline Dual operator+(double scalar, const Dual& x) {
    return Dual(scalar + x.val, x.deriv);
}

inline Dual operator-(double scalar, const Dual& x) {
    return Dual(scalar - x.val, -x.deriv);
}

inline Dual operator*(double scalar, const Dual& x) {
    return Dual(scalar * x.val, scalar * x.deriv);
}

inline Dual operator/(double scalar, const Dual& x) {
    double inv = 1.0 / x.val;
    return Dual(scalar * inv, -scalar * x.deriv * inv * inv);
}

// Comparison operators (operate on primal values only)
inline bool operator<(const Dual& a, const Dual& b) { return a.val < b.val; }
inline bool operator<=(const Dual& a, const Dual& b) { return a.val <= b.val; }
inline bool operator>(const Dual& a, const Dual& b) { return a.val > b.val; }
inline bool operator>=(const Dual& a, const Dual& b) { return a.val >= b.val; }
inline bool operator==(const Dual& a, const Dual& b) { return a.val == b.val; }
inline bool operator!=(const Dual& a, const Dual& b) { return a.val != b.val; }

inline bool operator<(const Dual& a, double b) { return a.val < b; }
inline bool operator<=(const Dual& a, double b) { return a.val <= b; }
inline bool operator>(const Dual& a, double b) { return a.val > b; }
inline bool operator>=(const Dual& a, double b) { return a.val >= b; }
inline bool operator==(const Dual& a, double b) { return a.val == b; }
inline bool operator!=(const Dual& a, double b) { return a.val != b; }

inline bool operator<(double a, const Dual& b) { return a < b.val; }
inline bool operator<=(double a, const Dual& b) { return a <= b.val; }
inline bool operator>(double a, const Dual& b) { return a > b.val; }
inline bool operator>=(double a, const Dual& b) { return a >= b.val; }
inline bool operator==(double a, const Dual& b) { return a == b.val; }
inline bool operator!=(double a, const Dual& b) { return a != b.val; }

// BVP Residual evaluation (legacy double version)
void compute_bvp_residual(
    const double mL[NUM_ACT_SET][3],
    const double nL[NUM_ACT_SET][3],
    const CRMShootingMethodParams& params,
    const double xf[NUM_STATES],
    double residual[NUM_ACT_SET * 6]
);

// Compute BVP Jacobian blocks using strictly analytic methods (no numerical differencing)
// Uses forward-mode AD with Dual numbers and analytic formulas
// J_yy = ∂r/∂(mL,nL)  : (6N × 6N)
// J_yu = ∂r/∂u       : (6N × 3N)
void compute_bvp_jacobians_fmad(
    const double mL[NUM_ACT_SET][3],
    const double nL[NUM_ACT_SET][3],
    const double u[NUM_ACT_SET][3],
    const CRMForwardKinematicsData& params,
    const double xf[NUM_STATES],
    double L_inserted,
    double dt,
    const double x_coil[NUM_ACT_SET][18],
    Eigen::MatrixXd& J_yy,  // Output: 6N × 6N
    Eigen::MatrixXd& J_yu   // Output: 6N × 3N
);

// Compute full BVP Jacobian blocks including state dependencies (A3.5)
// Extends compute_bvp_jacobians_fmad with J_yx and J_yxf
// J_yy = ∂r/∂(mL,nL)  : (6N × 6N)
// J_yu = ∂r/∂u        : (6N × 3N)
// J_yx = ∂r/∂x_t      : (6N × (18N+15)) - full state Jacobian
void compute_bvp_jacobians_full_analytic(
    const double mL[NUM_ACT_SET][3],
    const double nL[NUM_ACT_SET][3],
    const double u[NUM_ACT_SET][3],
    const CRMForwardKinematicsData& params,
    const double xf[NUM_STATES],
    double L_inserted,
    double dt,
    const double x_coil[NUM_ACT_SET][18],
    Eigen::MatrixXd& J_yy,  // Output: 6N × 6N
    Eigen::MatrixXd& J_yu,  // Output: 6N × 3N
    Eigen::MatrixXd& J_yx   // Output: 6N × (18N+15)
);

} // namespace CRMCatheterModel

#endif // CRM_BVP_JACOBIAN_HPP
