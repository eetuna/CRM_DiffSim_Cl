#pragma once

#include <cmath>
#include <type_traits>

namespace CRMCatheterModel {

// ... (previous templates kept as is) ...

// ============================================================================
// Mixed-type overloads for forward-mode AD (double × T → T)
// Enabled only when T != double to avoid ambiguity with fully templated versions
// ============================================================================

// Matrix multiplication C=A*B where A is double, B is T, C is T
template <typename T, int D1, int D2, int D3, typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mMult_AB_T(const double in_A[D1 * D2], const T in_B[D2 * D3], T out_C[D1 * D3]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D3; j++) {
			T sum = T(0.0);
			for (int k = 0; k < D2; k++) {
				sum += in_A[i * D2 + k] * in_B[k * D3 + j];
			}
			out_C[i * D3 + j] = sum;
		}
	}
}

// Matrix multiplication C=A^T*B where A is double, B is double, C is T
template <typename T, int D1, int D2, int D3, typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mMult_ATB_T(const double in_A[D1 * D2], const double in_B[D1 * D3], T out_C[D2 * D3]) {
	for (int i = 0; i < D2; i++) {
		for (int j = 0; j < D3; j++) {
			T sum = T(0.0);
			for (int k = 0; k < D1; k++) {
				sum += in_A[k * D2 + i] * in_B[k * D3 + j];
			}
			out_C[i * D3 + j] = sum;
		}
	}
}

// Matrix subtraction X=A-B where A is double, B is T, X is T
template <typename T, int D1, int D2, typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mSub_AB_T(const double in_A[D1 * D2], const T in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1 * D2; i++) {
		out_X[i] = in_A[i] - in_B[i];
	}
}

// Matrix subtraction X=A-B where A is T, B is double, X is T
template <typename T, int D1, int D2, typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mSub_AB_T(const T in_A[D1 * D2], const double in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1 * D2; i++) {
		out_X[i] = in_A[i] - in_B[i];
	}
}

// Matrix addition X=A+B where A is double, B is T, X is T
template <typename T, int D1, int D2, typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mAdd_AB_T(const double in_A[D1 * D2], const T in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1 * D2; i++) {
		out_X[i] = in_A[i] + in_B[i];
	}
}

// Matrix multiplication C=A*B where A is double, B is double, C is T (for type conversion)
template <typename T, int D1, int D2, int D3, typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mMult_AB_T(const double in_A[D1 * D2], const double in_B[D2 * D3], T out_C[D1 * D3]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D3; j++) {
			T sum = T(0.0);
			for (int k = 0; k < D2; k++) {
				sum += in_A[i * D2 + k] * in_B[k * D3 + j];
			}
			out_C[i * D3 + j] = sum;
		}
	}
}

} // namespace CRMCatheterModel
