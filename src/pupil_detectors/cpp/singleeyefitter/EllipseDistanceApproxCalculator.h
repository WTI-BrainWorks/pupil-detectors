#ifndef singleeyefitter_ellipsedistanceapproxcalculator_h__
#define singleeyefitter_ellipsedistanceapproxcalculator_h__

#include "mathHelper.h"

// Calculates:
//     r * (1 - ||A(p - t)||)
//
//          ||A(p - t)||   maps the ellipse to a unit circle
//      1 - ||A(p - t)||   measures signed distance from unit circle edge
// r * (1 - ||A(p - t)||)  scales this to major radius of ellipse, for (roughly) pixel distance
//
// Actually use (r - ||rAp - rAt||) and precalculate r, rA and rAt.

namespace singleeyefitter {

    using math::norm;

    template<typename T>
    class EllipseDistCalculator {
        public:
            EllipseDistCalculator(const Ellipse2D<T>& ellipse) : r(ellipse.major_radius)
            {
                using std::sin;
                using std::cos;
                rA << r* cos(ellipse.angle) / ellipse.major_radius, r* sin(ellipse.angle) / ellipse.major_radius,
                -r* sin(ellipse.angle) / ellipse.minor_radius, r* cos(ellipse.angle) / ellipse.minor_radius;
                rAt = rA * ellipse.center;
            }
            template<typename U>
            T operator()(U&& x, U&& y)
            {
                return calculate(std::forward<U>(x), std::forward<U>(y));
            }

            template<typename U>
            T calculate(U&& x, U&& y)
            {
                T rAxt((rA(0, 0) * x + rA(0, 1) * y) - rAt[0]);
                T rAyt((rA(1, 0) * x + rA(1, 1) * y) - rAt[1]);
                T xy_dist = norm(rAxt, rAyt);
                return (r - xy_dist);
            }

            // Squared mapped distance ||rA(p - t)||^2 -- avoids the sqrt in
            // norm(). The signed distance is r - sqrt(norm2), so the common test
            // |distance| <= thresh can be done on norm2 without a sqrt (see
            // within()), which is the hot loop in ellipse_true_support.
            template<typename U>
            T norm2(U&& x, U&& y) const
            {
                T rAxt((rA(0, 0) * x + rA(0, 1) * y) - rAt[0]);
                T rAyt((rA(1, 0) * x + rA(1, 1) * y) - rAt[1]);
                return rAxt * rAxt + rAyt * rAyt;
            }

            // True iff |r - ||rA(p - t)|| | <= thresh, without computing a sqrt.
            // |r - d| <= t  <=>  (r - t) <= d <= (r + t); square both sides
            // (d = sqrt(norm2) >= 0; the lower bound is automatic when r - t < 0).
            template<typename U>
            bool within(U&& x, U&& y, T thresh) const
            {
                T s = norm2(std::forward<U>(x), std::forward<U>(y));
                T hi = r + thresh;
                if (s > hi * hi)
                    return false;
                T lo = r - thresh;
                return lo <= T(0) || s >= lo * lo;
            }

            T radius() const { return r; }
        private:
            Eigen::Matrix<T, 2, 2> rA;
            Eigen::Matrix<T, 2, 1> rAt;
            T r;
    };

} //namespace

#endif //singleeyefitter_ellipsedistanceapproxcalculator_h__
