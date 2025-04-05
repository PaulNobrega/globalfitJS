// svd.js

/**
 * @fileoverview Provides linear algebra helper functions, including SVD decomposition,
 * matrix inversion using SVD, matrix multiplication, and transposition.
 * Designed for use with the lmFitGlobal fitting library.
 * Uses native JavaScript, adapted from Fortran numerical recipes logic.
 * Version: 1.0.0
 */

(function(global) {
    'use strict';

    /**
     * Solves the linear system Ax = b using Gaussian elimination with partial pivoting.
     * @param {number[][]} matrix - The matrix A (n x n).
     * @param {number[]} vector - The vector b (n).
     * @returns {number[]} The solution vector x (n).
     * @throws {Error} If the matrix is invalid, singular, or numerical issues occur.
     * @private
     */
    function solveLinearSystem(matrix, vector) {
        const n = vector.length;
        if (!matrix || matrix.length !== n || matrix.some(row => !row || row.length !== n)) {
            throw new Error(`Invalid matrix or vector size for solveLinearSystem. Matrix: ${matrix?.length}x${matrix?.[0]?.length}, Vector: ${n}`);
        }
        const augmentedMatrix = matrix.map((row, i) => [...row, vector[i]]);
        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmentedMatrix[k][i]) > Math.abs(augmentedMatrix[maxRow][i])) maxRow = k;
            }
            [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];
            if (Math.abs(augmentedMatrix[i][i]) < 1e-12) {
                let foundPivot = false;
                for(let k = i + 1; k < n; k++) {
                    if (Math.abs(augmentedMatrix[k][i]) >= 1e-12) {
                        [augmentedMatrix[i], augmentedMatrix[k]] = [augmentedMatrix[k], augmentedMatrix[i]];
                        foundPivot = true; break;
                    }
                }
                if (!foundPivot) throw new Error(`Matrix is singular or near-singular during Gaussian elimination at step ${i}.`);
            }
            for (let k = i + 1; k < n; k++) {
                const factor = augmentedMatrix[k][i] / augmentedMatrix[i][i];
                if (!isFinite(factor)) throw new Error(`Non-finite factor encountered during elimination at [${k},${i}]. Pivot: ${augmentedMatrix[i][i]}`);
                augmentedMatrix[k][i] = 0;
                for (let j = i + 1; j <= n; j++) augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
            }
        }
        const solution = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            if (Math.abs(augmentedMatrix[i][i]) < 1e-12) throw new Error(`Zero pivot encountered during back substitution at step ${i}.`);
            let sum = 0;
            for (let j = i + 1; j < n; j++) sum += augmentedMatrix[i][j] * solution[j];
            solution[i] = (augmentedMatrix[i][n] - sum) / augmentedMatrix[i][i];
            if (!isFinite(solution[i])) throw new Error(`Non-finite solution component encountered at index ${i}. Pivot: ${augmentedMatrix[i][i]}`);
        }
        return solution;
    }

    /**
     * Performs Singular Value Decomposition (SVD) on a matrix using a native JavaScript implementation.
     * Adapted from Fortran SVDCMP logic. Decomposes matrix A such that A = U * S * V^T. Assumes m >= n.
     * @param {number[][]} matrix - The input matrix A (m x n).
     * @returns {{u: number[][], s: number[], v: number[][]}} Object containing U (m x n), S (vector n), V (n x n).
     * @throws {Error} If convergence fails.
     * @private
     */
    function svdDecompose(matrix) {
        const a = matrix.map(row => [...row]);
        const m = a.length; if (m === 0) return { u: [], s: [], v: [] };
        const n = a[0].length; if (n === 0) return { u: a.map(() => []), s: [], v: [] };
        // Warning only, attempt to proceed if m < n
        if (m < n) console.warn("svdDecompose warning: Implementation assumes m >= n. Results may be incorrect.");

        const w = new Array(n).fill(0.0);
        const v = Array.from({ length: n }, () => new Array(n).fill(0.0));
        const rv1 = new Array(n).fill(0.0);
        let g = 0.0, scale = 0.0, anorm = 0.0;

        // Householder reduction
        for (let i = 0; i < n; i++) {
            scale = 0.0; for (let k = i; k < m; k++) scale += Math.abs(a[k][i]);
            if (scale !== 0.0) {
                let s = 0.0; for (let k = i; k < m; k++) { a[k][i] /= scale; s += a[k][i] * a[k][i]; }
                let f = a[i][i]; g = -Math.sign(f) * Math.sqrt(s); let h = f * g - s; a[i][i] = f - g;
                for (let j = i + 1; j < n; j++) { s = 0.0; for (let k = i; k < m; k++) s += a[k][i] * a[k][j]; f = s / h; for (let k = i; k < m; k++) a[k][j] += f * a[k][i]; }
                for (let k = i; k < m; k++) a[k][i] *= scale;
            }
            w[i] = scale * g; g = 0.0; scale = 0.0;
            if (i < n - 1) {
                for (let k = i + 1; k < n; k++) scale += Math.abs(a[i][k]);
                if (scale !== 0.0) {
                    let s = 0.0; for (let k = i + 1; k < n; k++) { a[i][k] /= scale; s += a[i][k] * a[i][k]; }
                    let f = a[i][i + 1]; g = -Math.sign(f) * Math.sqrt(s); let h = f * g - s; a[i][i + 1] = f - g;
                    for (let k = i + 1; k < n; k++) rv1[k] = a[i][k] / h;
                    for (let j = i + 1; j < m; j++) { s = 0.0; for (let k = i + 1; k < n; k++) s += a[j][k] * a[i][k]; for (let k = i + 1; k < n; k++) a[j][k] += s * rv1[k]; }
                    for (let k = i + 1; k < n; k++) a[i][k] *= scale;
                }
            }
            anorm = Math.max(anorm, (Math.abs(w[i]) + Math.abs(g))); rv1[i] = g;
        }
        // Accumulation of V
        for (let i = n - 1; i >= 0; i--) {
            if (i < n - 1) { if (rv1[i] !== 0.0) { for (let j = i + 1; j < n; j++) { let s = 0.0; for (let k = i + 1; k < n; k++) s += a[i][k] * v[k][j]; s = (s / a[i][i + 1]) / rv1[i]; for (let k = i + 1; k < n; k++) v[k][j] += s * a[i][k]; } } }
            for (let j = i + 1; j < n; j++) { v[i][j] = 0.0; v[j][i] = 0.0; } v[i][i] = 1.0;
        }
        // Accumulation of U (explicitly)
        const u = Array.from({ length: m }, () => new Array(n).fill(0.0));
        for (let i = n - 1; i >= 0; i--) {
            let l = i + 1; g = w[i]; for (let j = l; j < n; j++) a[i][j] = 0.0;
            if (g !== 0.0) { g = 1.0 / g; for (let j = l; j < n; j++) { let s = 0.0; for (let k = l; k < m; k++) s += a[k][i] * a[k][j]; let f = (s / a[i][i]) * g; for (let k = i; k < m; k++) a[k][j] += f * a[k][i]; } for (let j = i; j < m; j++) a[j][i] *= g; }
            else { for (let j = i; j < m; j++) a[j][i] = 0.0; } a[i][i] += 1.0;
        }
        for (let i = 0; i < m; i++) { for (let j = 0; j < n; j++) u[i][j] = a[i][j]; }
        // Diagonalization (QR)
        const maxIterations = 30 * n;
        for (let k = n - 1; k >= 0; k--) {
            for (let its = 0; its < maxIterations; its++) {
                let flag = true; let l = k;
                for (; l >= 0; l--) { if (l === 0 || Math.abs(rv1[l-1]) + anorm === anorm) { flag = false; break; } if (Math.abs(w[l - 1]) + anorm === anorm) break; }
                let c = 0.0; let s = 1.0;
                if (flag) { if (l > 0) { for (let i = l; i <= k; i++) { let f = s * rv1[i]; rv1[i] = c * rv1[i]; if (Math.abs(f) + anorm === anorm) break; g = w[i]; let h = Math.hypot(f, g); w[i] = h; h = 1.0 / h; c = g * h; s = -f * h; for (let j = 0; j < m; j++) { let y = u[j][l - 1]; let z = u[j][i]; u[j][l - 1] = y * c + z * s; u[j][i] = z * c - y * s; } } } }
                let z = w[k];
                if (l === k) { if (z < 0.0) { w[k] = -z; for (let j = 0; j < n; j++) v[j][k] = -v[j][k]; } break; }
                if (its === maxIterations - 1) { throw new Error("svdDecompose error: No convergence after maximum iterations."); }
                let x = w[l]; let y = w[k - 1]; g = rv1[k - 1]; let h = rv1[k]; let f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = Math.hypot(f, 1.0); let f_sign = (f >= 0.0) ? 1.0 : -1.0; f = ((x - z) * (x + z) + h * ((y / (f + f_sign * g)) - h)) / x;
                c = 1.0; s = 1.0;
                for (let j = l; j < k; j++) {
                    let i = j + 1; g = rv1[i]; y = w[i]; h = s * g; g = c * g; z = Math.hypot(f, h); rv1[j] = z; c = f / z; s = h / z;
                    f = x * c + g * s; g = g * c - x * s; h = y * s; y = y * c;
                    for (let jj = 0; jj < n; jj++) { x = v[jj][j]; z = v[jj][i]; v[jj][j] = x * c + z * s; v[jj][i] = z * c - x * s; }
                    z = Math.hypot(f, h); w[j] = z; if (z !== 0.0) { z = 1.0 / z; c = f * z; s = h * z; }
                    f = c * g + s * y; x = c * y - s * g;
                    for (let jj = 0; jj < m; jj++) { y = u[jj][j]; z = u[jj][i]; u[jj][j] = y * c + z * s; u[jj][i] = z * c - y * s; }
                } rv1[l] = 0.0; rv1[k] = f; w[k] = x;
            }
        }
        return { u: u, s: w, v: v };
    }

    /**
     * Multiplies two matrices A and B.
     * @param {number[][]} a - Matrix A.
     * @param {number[][]} b - Matrix B.
     * @returns {number[][]} The resulting matrix A * B.
     * @throws {Error} If dimensions mismatch or elements are invalid.
     * @private
     */
    function multiplyMatrices(a, b) {
        const aRows = a.length; if (aRows === 0) return []; const aCols = a[0].length;
        const bRows = b.length; if (bRows === 0) return a.map(row => []); const bCols = b[0].length;
        if (aCols !== bRows) throw new Error(`Matrix dimension mismatch: A(${aRows}x${aCols}), B(${bRows}x${bCols})`);
        const result = new Array(aRows);
        for (let i = 0; i < aRows; i++) {
            result[i] = new Array(bCols).fill(0);
            for (let j = 0; j < bCols; j++) {
                for (let k = 0; k < aCols; k++) {
                    if (a[i]?.[k] === undefined || b[k]?.[j] === undefined) throw new Error(`Undefined element accessed: A[${i}][${k}] or B[${k}][${j}]`);
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        } return result;
    }

    /**
     * Transposes a matrix.
     * @param {number[][]} matrix - The matrix to transpose.
     * @returns {number[][]} The transposed matrix.
     * @throws {Error} If elements are invalid.
     * @private
     */
    function transposeMatrix(matrix) {
        const rows = matrix.length; if (rows === 0) return []; const cols = matrix[0].length; if (cols === 0) return matrix.map(() => []);
        const result = new Array(cols);
        for (let j = 0; j < cols; j++) {
            result[j] = new Array(rows);
            for (let i = 0; i < rows; i++) {
                if (matrix[i]?.[j] === undefined) throw new Error(`Undefined element accessed during transpose at [${i}][${j}]`);
                result[j][i] = matrix[i][j];
            }
        } return result;
    }

    /**
     * Calculates the pseudo-inverse of a square matrix using SVD.
     * @param {number[][]} matrix - The square matrix to invert.
     * @returns {number[][]} The pseudo-inverse matrix.
     * @throws {Error} If matrix is invalid or SVD fails.
     * @private
     */
    function invertMatrixUsingSVD(matrix) {
        const n = matrix.length; if (n === 0 || matrix.some(row => !row || row.length !== n)) throw new Error("Matrix must be square for SVD inversion.");
        const { u, s, v } = svdDecompose(matrix);
        if (!u || u.length === 0 || !s || s.length === 0 || !v || v.length === 0) throw new Error("SVD decomposition failed or returned empty matrices.");
        const uCols = u[0]?.length, sLen = s.length, vRows = v.length, vCols = v[0]?.length;
        // SVD of NxN matrix should yield U(MxN), S(N), V(NxN) where M>=N. Here M=N.
        if (uCols !== n || sLen !== n || vRows !== n || vCols !== n) console.warn(`SVD returned unexpected dimensions: U(${u.length}x${uCols}), S(${sLen}), V(${vRows}x${vCols}) for input NxN (${n}x${n})`);
        const tolerance = 1e-10;
        const sInv = new Array(n).fill(0).map(() => new Array(n).fill(0));
        const numSingularValues = Math.min(n, s.length);
        for (let i = 0; i < numSingularValues; i++) { if (Math.abs(s[i]) > tolerance) sInv[i][i] = 1 / s[i]; else sInv[i][i] = 0; }
        const vMatrix = v; const uTranspose = transposeMatrix(u);
        if (uTranspose.length !== n) throw new Error(`Dimension mismatch for U transpose: Expected ${n} rows, got ${uTranspose.length}`);
        const vsInv = multiplyMatrices(vMatrix, sInv); const inverse = multiplyMatrices(vsInv, uTranspose);
        if (inverse.length !== n || inverse[0]?.length !== n) console.warn(`SVD inverse resulted in non-square matrix (${inverse.length}x${inverse[0]?.length}) for square input (${n}x${n}).`);
        return inverse;
    }

    // Expose necessary functions to the global scope under a namespace
    global.LM_SVD_HELPERS = {
        solveLinearSystem,
        svdDecompose,
        multiplyMatrices,
        transposeMatrix,
        invertMatrixUsingSVD
    };

// Establish the root object, `window` in the browser, or `global` on the server.
})(typeof window !== 'undefined' ? window : global);