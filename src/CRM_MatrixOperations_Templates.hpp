#pragma once

#include <cmath>
#include <type_traits>

namespace CRMCatheterModel {

// Templated matrix operations for automatic differentiation
// All matrices are stored as one-dimensional arrays in row-major order
// These templated versions support generic scalar types (double, Dual, etc.)

// Matrix multiplication C=A*B  ( A:D1xD2 B:D2xD3 gives C:D1xD3 )
template <typename T, int D1, int D2, int D3>
void mMult_AB_T(const T in_A[D1 * D2], const T in_B[D2 * D3], T out_C[D1 * D3]) {
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

// Matrix multiplication C=A^T*B  (A transposed times B)  ( A:D1xD2, A^T:D2xD1, B:D1xD3 gives C:D2xD3 )
template <typename T, int D1, int D2, int D3>
void mMult_ATB_T(const T in_A[D1 * D2], const T in_B[D1 * D3], T out_C[D2 * D3]) {
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

// Matrix multiplication C=A*B^T  (A times B transposed)  ( A:D1xD2, B:D3xD2, B^T:D2xD3,  gives C:D1xD3 )
template <typename T, int D1, int D2, int D3>
void mMult_ABT_T(const T in_A[D1 * D2], const T in_B[D3 * D2], T out_C[D1 * D3]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D3; j++) {
			T sum = T(0.0);
			for (int k = 0; k < D2; k++) {
				sum += in_A[i * D2 + k] * in_B[j * D2 + k];
			}
			out_C[i * D3 + j] = sum;
		}
	}
}

// Matrix scalar multiplication C=s*A  ( s:scalar, A,C:D1xD2 )
template <typename T, int D1, int D2>
void mMult_sA_T(const T s, const T in_A[D1 * D2], T out_C[D1 * D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			out_C[i * D2 + j] = s * in_A[i * D2 + j];
		}
	}
}

// Matrix multiply and add C=C+A*B  ( A:D1xD2 B:D2xD3 gives C:D1xD3 )
template <typename T, int D1, int D2, int D3>
void mMultAdd_AB_T(const T in_A[D1 * D2], const T in_B[D2 * D3], T out_C[D1 * D3]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D3; j++) {
			T sum = T(0.0);
			for (int k = 0; k < D2; k++) {
				sum += in_A[i * D2 + k] * in_B[k * D3 + j];
			}
			out_C[i * D3 + j] += sum;
		}
	}
}

// Matrix multiply and subtract C=C-A*B  ( A:D1xD2 B:D2xD3 gives C:D1xD3 )
template <typename T, int D1, int D2, int D3>
void mMultSub_AB_T(const T in_A[D1 * D2], const T in_B[D2 * D3], T out_C[D1 * D3]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D3; j++) {
			T sum = T(0.0);
			for (int k = 0; k < D2; k++) {
				sum += in_A[i * D2 + k] * in_B[k * D3 + j];
			}
			out_C[i * D3 + j] -= sum;
		}
	}
}

// Matrix addition X=A+B  ( A,B,X:D1xD2 )
template <typename T, int D1, int D2>
void mAdd_AB_T(const T in_A[D1 * D2], const T in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			out_X[i * D2 + j] = in_A[i * D2 + j] + in_B[i * D2 + j];
		}
	}
}

// Matrix addition A=A+B  ( A,B:D1xD2 )
template <typename T, int D1, int D2>
void mAdd_AB_T(T inout_A[D1 * D2], const T in_B[D1 * D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			inout_A[i * D2 + j] += in_B[i * D2 + j];
		}
	}
}

// Matrix addition X=A+s*B  ( A,B,X:D1xD2, s:scalar )
template <typename T, int D1, int D2>
void mAdd_AsB_T(const T in_A[D1 * D2], const T in_s, const T in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			out_X[i * D2 + j] = in_A[i * D2 + j] + in_s * in_B[i * D2 + j];
		}
	}
}

// Matrix addition X=A+B+C  ( A,B,C,X:D1xD2 )
template <typename T, int D1, int D2>
void mAdd_ABC_T(const T in_A[D1 * D2], const T in_B[D1 * D2], const T in_C[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			out_X[i * D2 + j] = in_A[i * D2 + j] + in_B[i * D2 + j] + in_C[i * D2 + j];
		}
	}
}

// Matrix subtraction X=A-B   ( A,B,X:D1xD2 )
template <typename T, int D1, int D2>
void mSub_AB_T(const T in_A[D1 * D2], const T in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			out_X[i * D2 + j] = in_A[i * D2 + j] - in_B[i * D2 + j];
		}
	}
}

// Matrix subtraction A=A-B   ( A,B:D1xD2 )
template <typename T, int D1, int D2>
void mSub_AB_T(T inout_A[D1 * D2], const T in_B[D1 * D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			inout_A[i * D2 + j] -= in_B[i * D2 + j];
		}
	}
}

// 2-norm squared of a vector  ( D1x1 vector )
template <typename T, int D1>
T vNormSq_T(const T v[D1]) {
	T vn = T(0.0);
	for (int i = 0; i < D1; i++) {
		vn += v[i] * v[i];
	}
	return vn;
}

// Copy matrix: B=A  (D1xD2 matrices with 2D indexing)
template <typename T, int D1, int D2>
void mCopy_ABm_T(const T in_A[D1][D2], T out_B[D1][D2]) {
	for (int i = 0; i < D1; i++) {
		for (int j = 0; j < D2; j++) {
			out_B[i][j] = in_A[i][j];
		}
	}
}

// Copy vector: B=A  (D1x1 vector)
template <typename T, int D1>
void mCopy_AB_T(const T in_A[D1], T out_B[D1]) {
	for (int i = 0; i < D1; i++) {
		out_B[i] = in_A[i];
	}
}

// Skew-symmetric matrix from 3D vector (cross product operator)
// Constructs [w]_× from w such that [w]_× v = w × v
// what is stored as a 1-dim array in row major order
template <typename T>
void wHat_T(const T in_w[3], T out_what[9]) {
	out_what[0] = T(0.0);
	out_what[1] = -in_w[2];
	out_what[2] = in_w[1];
	out_what[3] = in_w[2];
	out_what[4] = T(0.0);
	out_what[5] = -in_w[0];
	out_what[6] = -in_w[1];
	out_what[7] = in_w[0];
	out_what[8] = T(0.0);
}

// Skew-symmetric matrix from 3D vector with stride
template <typename T>
void wHat_T(const T in_w[3], T out_what[9], unsigned int stride) {
	out_what[0] = T(0.0);
	out_what[1] = -in_w[2 * stride];
	out_what[2] = in_w[1 * stride];
	out_what[3] = in_w[2 * stride];
	out_what[4] = T(0.0);
	out_what[5] = -in_w[0 * stride];
	out_what[6] = -in_w[1 * stride];
	out_what[7] = in_w[0 * stride];
	out_what[8] = T(0.0);
}

// ============================================================================
// Mixed-type overloads for forward-mode AD (double × T → T)
// ============================================================================

// Matrix multiplication C=A*B where A is double, B is T, C is T
template <typename T, int D1, int D2, int D3,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
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
template <typename T, int D1, int D2, int D3,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
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
template <typename T, int D1, int D2,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mSub_AB_T(const double in_A[D1 * D2], const T in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1 * D2; i++) {
		out_X[i] = in_A[i] - in_B[i];
	}
}

// Matrix subtraction X=A-B where A is T, B is double, X is T
template <typename T, int D1, int D2,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mSub_AB_T(const T in_A[D1 * D2], const double in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1 * D2; i++) {
		out_X[i] = in_A[i] - in_B[i];
	}
}

// Matrix addition X=A+B where A is double, B is T, X is T
template <typename T, int D1, int D2,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
void mAdd_AB_T(const double in_A[D1 * D2], const T in_B[D1 * D2], T out_X[D1 * D2]) {
	for (int i = 0; i < D1 * D2; i++) {
		out_X[i] = in_A[i] + in_B[i];
	}
}

// Matrix multiplication C=A*B where A is double, B is double, C is T (for type conversion)
template <typename T, int D1, int D2, int D3,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
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
