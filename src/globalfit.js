/**
 * @fileoverview Provides Levenberg-Marquardt fitting functions:
 *   - lmFitGlobal: For simultaneous fitting of multiple datasets with linking/fixing.
 *   - lmFit: A wrapper for fitting a single dataset using lmFitGlobal.
 *   - lmFitIndependent: Sequentially fits multiple datasets independently using lmFitGlobal.
 * Includes helpers for parameter mapping, chi-squared calculation, Jacobian calculation,
 * constraint application, and statistics calculation.
 * Depends on svd.js for linear algebra operations.
 * Version: 1.2.5 (Uses fixMap-style linkMap, adds covariance regularization, uses Math.abs for StdErr)
 */

(function(global) {
    'use strict';

    // Check if SVD helpers are loaded from svd.js
    if (!global.LM_SVD_HELPERS) {
        throw new Error("Error: svd.js must be loaded before fit.js");
    }
    // Import helpers from the global object created by svd.js
    const {
        solveLinearSystem,
        invertMatrixUsingSVD
    } = global.LM_SVD_HELPERS;


    // ============================================================================
    // Fitting Algorithm Helpers (Internal)
    // ============================================================================

    /** Log levels for controlling verbosity */
    const LOG_LEVELS = { none: 0, error: 1, warn: 2, info: 3, debug: 4 };

    /**
     * Calculates the gradient vector (J^T * r).
     * @private
     */
    function calculateGradient(jacobian, residuals) {
        const numPoints = jacobian.length; if (numPoints === 0) return []; const numParams = jacobian[0]?.length ?? 0;
        if (numParams === 0 && residuals.length === 0) return []; if (numParams === 0 && residuals.length > 0) throw new Error("Jacobian has zero columns but residuals exist.");
        if (numPoints !== residuals.length) throw new Error(`Jacobian rows (${numPoints}) must match residuals length (${residuals.length}).`);
        const gradient = new Array(numParams).fill(0);
        for (let j = 0; j < numParams; j++) {
            for (let i = 0; i < numPoints; i++) {
                if (jacobian[i]?.[j] === undefined || !isFinite(jacobian[i][j]) || residuals[i] === undefined || !isFinite(residuals[i])) throw new Error(`Invalid value in gradient calculation at point ${i}, param ${j}.`);
                gradient[j] += jacobian[i][j] * residuals[i];
            }
        } return gradient;
    }

    /**
     * Calculates the approximate Hessian matrix (J^T * J).
     * @private
     */
    function calculateHessian(jacobian) {
        const numPoints = jacobian.length; if (numPoints === 0) return []; const numParams = jacobian[0]?.length ?? 0; if (numParams === 0) return [];
        const hessian = new Array(numParams).fill(0).map(() => new Array(numParams).fill(0));
        for (let j = 0; j < numParams; j++) {
            for (let k = j; k < numParams; k++) {
                let sum = 0;
                for (let i = 0; i < numPoints; i++) {
                    if (jacobian[i]?.[j] === undefined || !isFinite(jacobian[i][j]) || jacobian[i]?.[k] === undefined || !isFinite(jacobian[i][k])) throw new Error(`Invalid value in Hessian calculation at point ${i}, params ${j}, ${k}.`);
                    sum += jacobian[i][j] * jacobian[i][k];
                }
                hessian[j][k] = sum; if (j !== k) hessian[k][j] = sum;
            }
        } return hessian;
    }

    // ============================================================================
    // *** MAPPING LOGIC (v1.4.3 style - Corrected for new linkMap) ***
    // ============================================================================

    /**
     * Creates mapping between full parameter structure and the flat array of active parameters.
     * Internal helper function for lmFitGlobal.
     * @param {number[][][]} initialParameters
     * @param {(number|string|null)[][][]} linkMapInput - NEW: Nested array matching initialParameters, values are group IDs or null.
     * @param {boolean[][][]} fixMapInput
     * @param {Function} logFn
     * @private
     */
    function setupParameterMapping(initialParameters, linkMapInput, fixMapInput, logFn) {
        logFn("--- Running setupParameterMapping (New linkMap structure v1.4.3 logic) ---", 'debug');
        logFn(`Received linkMapInput (structure like fixMap): ${linkMapInput ? 'Provided' : 'Null'}`, 'debug');
        logFn(`Received fixMapInput: ${JSON.stringify(fixMapInput)}`, 'debug');

        // --- 1. Flatten Parameters and Create Coordinate Mapping ---
        const paramStructure = [];
        const flatInitialParams = [];
        const paramCoordinates = []; // Stores [[dsIdx, pIdx], vIdx] for each flat index
        let currentFlatIndex = 0;
        const fixMap = fixMapInput ? JSON.parse(JSON.stringify(fixMapInput)) : []; // Deep copy or empty
        const linkMap = linkMapInput ? JSON.parse(JSON.stringify(linkMapInput)) : null; // Deep copy or null

        initialParameters.forEach((dsParams, dsIdx) => {
            paramStructure.push([]);
            if (!fixMap[dsIdx]) fixMap[dsIdx] = [];
            dsParams.forEach((pArray, pIdx) => {
                paramStructure[dsIdx].push(pArray.length);
                if (!fixMap[dsIdx][pIdx]) fixMap[dsIdx][pIdx] = new Array(pArray.length).fill(false);
                pArray.forEach((pValue, vIdx) => {
                    flatInitialParams.push(pValue);
                    // Store coordinates based on the ORIGINAL initialParameters structure
                    paramCoordinates.push([[dsIdx, pIdx], vIdx]);
                    if (fixMap[dsIdx][pIdx].length <= vIdx) fixMap[dsIdx][pIdx][vIdx] = false;
                    currentFlatIndex++;
                });
            });
        });

        const nTotalParams = flatInitialParams.length;
        const masterMap = new Array(nTotalParams).fill(-1); // Ensure initialized
        const isFixed = new Array(nTotalParams).fill(false); // Ensure initialized
        const activeParamInfo = []; const activeInitialParams = []; const activeParamLabels = [];

        // --- 2. Apply fixMap ---
        // Use paramCoordinates which maps flatIdx -> original coordinate
        paramCoordinates.forEach((coord, flatIdx) => {
            const [[dsIdx, paramIdx], valIdx] = coord;
            if (fixMap[dsIdx]?.[paramIdx]?.[valIdx] === true) {
                isFixed[flatIdx] = true;
            }
        });
        logFn(`Initial isFixed array (after fixMap): ${JSON.stringify(isFixed)}`, 'debug');

        // --- 3. Process linkMap (NEW STRUCTURE LOGIC - v1.4.3 style) ---
        const linkGroupsById = {}; // { groupId: [flatIdx1, flatIdx2, ...], ... }

        if (linkMap) {
            // Iterate through the structure of linkMapInput, which should match the structure of initialParameters
            linkMap.forEach((dsLinkMap, dsLinkIdx) => {
                if (!dsLinkMap || dsLinkIdx >= initialParameters.length) return; // Skip if dataset doesn't exist in params
                dsLinkMap.forEach((modelLinkMap, modelLinkIdx) => {
                    if (!modelLinkMap || modelLinkIdx >= initialParameters[dsLinkIdx].length) return; // Skip if model group doesn't exist
                    modelLinkMap.forEach((groupId, paramLinkIdx) => {
                         if (paramLinkIdx >= initialParameters[dsLinkIdx][modelLinkIdx].length) return; // Skip if param index is out of bounds

                        if (groupId !== null && groupId !== undefined && groupId !== '') {
                            // Find the corresponding flatIndex using paramCoordinates by matching coordinates
                            const targetCoord = [dsLinkIdx, modelLinkIdx, paramLinkIdx];
                            const flatIdx = paramCoordinates.findIndex(pc =>
                                pc[0][0] === targetCoord[0] &&
                                pc[0][1] === targetCoord[1] &&
                                pc[1] === targetCoord[2]
                            );

                            if (flatIdx !== -1) {
                                if (!linkGroupsById[groupId]) {
                                    linkGroupsById[groupId] = [];
                                }
                                linkGroupsById[groupId].push(flatIdx);
                            } else {
                                // This case should ideally not happen if initialParameters and linkMapInput are consistent
                                logFn(`Could not find flat index for link coordinate [${targetCoord.join(', ')}]`, 'warn');
                            }
                        }
                    });
                });
            });
            logFn(`Collected Link Groups by ID: ${JSON.stringify(linkGroupsById)}`, 'debug'); // Log collected groups

            // Process collected groups
            Object.entries(linkGroupsById).forEach(([groupId, flatIndices]) => {
                if (flatIndices.length < 2) {
                    logFn(`Link Group ${groupId} has only one member (${flatIndices[0]}). Ignoring link.`, 'debug');
                    return;
                }
                logFn(`Processing Link Group ${groupId}: Flat indices ${JSON.stringify(flatIndices)}`, 'debug');

                let masterFlatIndex = -1;
                let masterCoord = null;
                let allInitiallyFixed = true;

                for (const flatIdx of flatIndices) {
                    if (flatIdx < 0 || flatIdx >= nTotalParams) { logFn(`  Invalid flat index ${flatIdx} found in link group ${groupId}. Skipping.`, 'warn'); continue; }
                    if (!isFixed[flatIdx]) {
                        allInitiallyFixed = false;
                        if (masterFlatIndex === -1) { masterFlatIndex = flatIdx; masterCoord = paramCoordinates[flatIdx]; }
                    }
                }

                if (allInitiallyFixed) {
                    masterFlatIndex = flatIndices[0];
                    if (masterFlatIndex < 0 || masterFlatIndex >= nTotalParams) { logFn(`  Invalid nominal master index ${masterFlatIndex} for fixed group ${groupId}. Skipping group.`, 'warn'); return; }
                    logFn(`  Link Group ${groupId} consists entirely of initially fixed parameters. Nominal master: flat index ${masterFlatIndex}`, 'debug');
                    if (!isFixed[masterFlatIndex]) {
                         logFn(`  Marking nominal master ${JSON.stringify(paramCoordinates[masterFlatIndex])} (flat ${masterFlatIndex}) as fixed because entire group is fixed.`, 'debug');
                         isFixed[masterFlatIndex] = true;
                    }
                    flatIndices.forEach(flatIdx => {
                         if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) {
                            masterMap[flatIdx] = masterFlatIndex;
                            flatInitialParams[flatIdx] = flatInitialParams[masterFlatIndex];
                            logFn(`    Linked fixed slave ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}) to fixed master ${masterFlatIndex}. State: isFixed=${isFixed[flatIdx]}, masterMap=${masterMap[flatIdx]}`, 'debug');
                        } else {
                             logFn(`    Nominal Master ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}). State: isFixed=${isFixed[flatIdx]}, masterMap=${masterMap[flatIdx]}`, 'debug');
                        }
                    });
                }
                else if (masterFlatIndex !== -1) {
                    logFn(`  Link Group ${groupId} master: ${JSON.stringify(masterCoord)} (flat index ${masterFlatIndex})`, 'debug');
                    logFn(`    Master ${JSON.stringify(paramCoordinates[masterFlatIndex])} (flat ${masterFlatIndex}). State: isFixed=${isFixed[masterFlatIndex]}, masterMap=${masterMap[masterFlatIndex]}`, 'debug');
                    flatIndices.forEach(flatIdx => {
                         if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) {
                            const initiallyFixed = isFixed[flatIdx];
                            if (isFixed[flatIdx]) {
                                logFn(`    Linking overrides fixed status for ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}).`, 'warn');
                                isFixed[flatIdx] = false;
                            }
                            masterMap[flatIdx] = masterFlatIndex;
                            flatInitialParams[flatIdx] = flatInitialParams[masterFlatIndex];
                            logFn(`    Linked slave ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}) to master ${masterFlatIndex}. InitialFixed=${initiallyFixed}. Final State: isFixed=${isFixed[flatIdx]}, masterMap=${masterMap[flatIdx]}`, 'debug');
                        }
                    });
                } else {
                     logFn(`  Internal Logic Error: Could not determine master type for group ${groupId}.`, 'error');
                }
            });
        } else {
            logFn("No linkMap provided or linkMap is null.", 'debug');
        }

        logFn(`Final isFixed array (after linking): ${JSON.stringify(isFixed)}`, 'debug');
        logFn(`Final masterMap array: ${JSON.stringify(masterMap)}`, 'debug');

        // --- 4. Identify active parameters ---
        let activeIndex = 0;
        for (let i = 0; i < nTotalParams; i++) {
            if (!isFixed[i] && masterMap[i] === -1) {
                activeParamInfo.push({ originalCoord: paramCoordinates[i], flatIndex: i, activeIndex: activeIndex });
                activeInitialParams.push(flatInitialParams[i]); const [[ds, p], v] = paramCoordinates[i]; activeParamLabels.push(`ds${ds}_p${p}_v${v}`); activeIndex++;
            }
        }
        const totalActiveParams = activeInitialParams.length;
        logFn(`Identified ${totalActiveParams} active parameters: ${JSON.stringify(activeParamLabels)}`, 'debug');
        logFn(`Active Initial Params: ${JSON.stringify(activeInitialParams)}`, 'debug');

        // --- 5. Create the reconstruction function ---
        const reconstructParams = (activeParamsCurrent) => {
            if (activeParamsCurrent.length !== totalActiveParams) throw new Error(`reconstructParams expects ${totalActiveParams} params, received ${activeParamsCurrent.length}`);
            const reconstructedFlat = [...flatInitialParams];
            activeParamInfo.forEach((info, actIdx) => { if (actIdx >= activeParamsCurrent.length) throw new Error(`Mismatch activeParamInfo/activeParams.`); reconstructedFlat[info.flatIndex] = activeParamsCurrent[actIdx]; });
            for (let i = 0; i < nTotalParams; i++) { if (masterMap[i] !== -1) { if (masterMap[i] < 0 || masterMap[i] >= reconstructedFlat.length) throw new Error(`Master index ${masterMap[i]} out of bounds.`); reconstructedFlat[i] = reconstructedFlat[masterMap[i]]; } }
            const nestedParams = []; let currentFlatIdx_recon = 0;
            paramStructure.forEach((dsStruct, dsIdx) => {
                nestedParams[dsIdx] = [];
                dsStruct.forEach((pLen, pIdx) => {
                    nestedParams[dsIdx][pIdx] = [];
                    for (let vIdx = 0; vIdx < pLen; vIdx++) { if (currentFlatIdx_recon >= reconstructedFlat.length) throw new Error(`Flat index ${currentFlatIdx_recon} out of bounds.`); nestedParams[dsIdx][pIdx][vIdx] = reconstructedFlat[currentFlatIdx_recon]; currentFlatIdx_recon++; }
                });
            });
            if (currentFlatIdx_recon !== nTotalParams) logFn(`Param reconstruction mismatch: ${currentFlatIdx_recon} vs ${nTotalParams}`, 'warn');
            return nestedParams;
        };

        return {
            activeInitialParams, reconstructParams, activeParamInfo, totalActiveParams,
            paramStructure, activeParamLabels,
            isFixed, masterMap, paramCoordinates, nTotalParams
        };
    } // <-- End of setupParameterMapping function (New Logic v1.4.3 style)


    /**
     * Calculates the global chi-squared value.
     * @private
     */
    function calculateChiSquaredGlobal(data, modelFunction, reconstructParamsFunc, activeParams, robustCostFunction, paramStructure, logFn) {
        const reconstructedParams = reconstructParamsFunc(activeParams); let chiSquared = 0; let totalPoints = 0;
        data.x.forEach((xDataset, dsIdx) => {
            const yDataset = data.y[dsIdx]; const yeDataset = data.ye[dsIdx]; const models = modelFunction[dsIdx]; const paramsForDs = reconstructedParams[dsIdx];
            if (!xDataset || !yDataset || !yeDataset || !models || !paramsForDs || xDataset.length !== yDataset.length || xDataset.length !== yeDataset.length) { logFn(`Data length/param mismatch dataset ${dsIdx}`, 'error'); return NaN; }
            for (let ptIdx = 0; ptIdx < xDataset.length; ptIdx++) {
                totalPoints++; const xPoint = xDataset[ptIdx]; const yPoint = yDataset[ptIdx]; const yePoint = yeDataset[ptIdx];
                if (yePoint === 0) { logFn(`Zero error point ${ptIdx} ds ${dsIdx}. Skipping.`, 'warn'); continue; }
                if (!isFinite(yPoint) || !isFinite(yePoint)) { logFn(`Non-finite y/ye point ${ptIdx} ds ${dsIdx}. Skipping.`, 'warn'); continue; }
                let combinedModelValue = 0;
                try {
                    models.forEach((modelFunc, paramIdx) => {
                        if (!paramsForDs[paramIdx]) throw new Error(`Missing params ds ${dsIdx}, pIdx ${paramIdx}.`);
                        const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]);
                        if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} invalid result x=${xPoint}: ${JSON.stringify(componentModelResult)}`);
                        combinedModelValue += componentModelResult[0];
                    });
                } catch (error) { logFn(`Error evaluating model ds ${dsIdx}, pt ${ptIdx}: ${error.message}`, 'error'); return NaN; }
                if (!isFinite(combinedModelValue)) { logFn(`Non-finite model value pt ${ptIdx} ds ${dsIdx}. Skipping.`, 'warn'); continue; }
                const residual = (yPoint - combinedModelValue) / yePoint;
                if (robustCostFunction === 1) chiSquared += Math.abs(residual);
                else if (robustCostFunction === 2) chiSquared += Math.log(1 + 0.5 * (residual * residual));
                else chiSquared += residual * residual;
            }
        });
        if (!isFinite(chiSquared)) { logFn("Non-finite chi-squared calculated.", 'error'); return Infinity; }
        return chiSquared;
    }

    /**
     * Calculates the global Jacobian matrix and residuals vector using numerical differentiation.
     * @private
     */
    function calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParamsFunc, activeParams, activeParamInfo, epsilon, paramStructure, logFn) {
        const reconstructedParams = reconstructParamsFunc(activeParams); const residuals = []; const jacobianRows = []; const totalActiveParams = activeParamInfo.length; let totalPoints = 0;
        data.x.forEach((xDataset, dsIdx) => {
            const yDataset = data.y[dsIdx]; const yeDataset = data.ye[dsIdx]; const models = modelFunction[dsIdx]; const paramsForDs = reconstructedParams[dsIdx];
            if (!xDataset || !yDataset || !yeDataset || !models || !paramsForDs || xDataset.length !== yDataset.length || xDataset.length !== yeDataset.length) throw new Error(`Data length mismatch ds ${dsIdx} Jacobian.`);
            for (let ptIdx = 0; ptIdx < xDataset.length; ptIdx++) {
                totalPoints++; const xPoint = xDataset[ptIdx]; const yPoint = yDataset[ptIdx]; const yePoint = yeDataset[ptIdx];
                if (yePoint === 0) throw new Error(`Zero error pt ${ptIdx} ds ${dsIdx} Jacobian.`);
                if (!isFinite(yPoint) || !isFinite(yePoint)) throw new Error(`Non-finite y/ye pt ${ptIdx} ds ${dsIdx} Jacobian.`);
                let originalCombinedModelValue = 0;
                try {
                    models.forEach((modelFunc, paramIdx) => {
                        if (!paramsForDs[paramIdx]) throw new Error(`Missing params ds ${dsIdx}, pIdx ${paramIdx} (Jacobian orig).`);
                        const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]);
                        if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} invalid result x=${xPoint}: ${JSON.stringify(componentModelResult)}`);
                        originalCombinedModelValue += componentModelResult[0];
                    });
                } catch (error) { logFn(`Error evaluating model (orig) ds ${dsIdx}, pt ${ptIdx}: ${error.message}`, 'error'); throw error; }
                if (!isFinite(originalCombinedModelValue)) throw new Error(`Non-finite orig model value pt ${ptIdx} ds ${dsIdx}.`);
                const residual = (yPoint - originalCombinedModelValue) / yePoint; residuals.push(residual);
                const jacobianRow = new Array(totalActiveParams).fill(0);
                for (let actIdx = 0; actIdx < totalActiveParams; actIdx++) {
                    const perturbedActiveParams = [...activeParams]; const originalValue = perturbedActiveParams[actIdx];
                    let h = epsilon * Math.abs(originalValue) + epsilon; if (h === 0) h = epsilon;
                    perturbedActiveParams[actIdx] += h; const perturbedReconstructed = reconstructParamsFunc(perturbedActiveParams);
                    let perturbedCombinedModelValue = 0; let perturbationFailed = false;
                    try {
                        const perturbedParamsForDs = perturbedReconstructed[dsIdx];
                        models.forEach((modelFunc, paramIdx) => {
                            if (perturbationFailed) return; if (!perturbedParamsForDs || !perturbedParamsForDs[paramIdx]) throw new Error(`Missing params ds ${dsIdx}, pIdx ${paramIdx} (Jacobian pert).`);
                            const componentParams = perturbedParamsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]);
                            if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) { logFn(`Model ${dsIdx}-${paramIdx} invalid result during perturbation x=${xPoint}`, 'warn'); perturbationFailed = true; return; }
                            perturbedCombinedModelValue += componentModelResult[0];
                        });
                    } catch (error) { logFn(`Error evaluating model (pert) ds ${dsIdx}, pt ${ptIdx}, actP ${actIdx}: ${error.message}`, 'error'); perturbationFailed = true; }
                    if (perturbationFailed || !isFinite(perturbedCombinedModelValue)) { logFn(`Non-finite/failed pert model value pt ${ptIdx}, ds ${dsIdx}, actP ${actIdx}. Deriv=0.`, 'warn'); jacobianRow[actIdx] = 0; }
                    else { const derivative = (perturbedCombinedModelValue - originalCombinedModelValue) / h; jacobianRow[actIdx] = -derivative / yePoint; }
                    if (!isFinite(jacobianRow[actIdx])) { logFn(`Non-finite Jacobian element pt ${ptIdx}, ds ${dsIdx}, actP ${actIdx}. Setting=0.`, 'warn'); jacobianRow[actIdx] = 0; }
                } jacobianRows.push(jacobianRow);
            }
        });
        if (residuals.length !== jacobianRows.length) throw new Error(`Internal error: Residuals (${residuals.length}) != Jacobian rows (${jacobianRows.length})`);
        if (jacobianRows.length > 0 && jacobianRows[0].length !== totalActiveParams) throw new Error(`Internal error: Jacobian cols (${jacobianRows[0].length}) != active params (${totalActiveParams})`);
        return { jacobian: jacobianRows, residuals };
    }

    /**
     * Applies constraints (box and custom function) to the reconstructed parameter structure.
     * @private
     */
    function applyConstraintsGlobal(reconstructedParams, constraints, activeParamInfo, paramStructure, constraintFunction, logFn) {
        let constrainedParams = reconstructedParams; let changedActive = false; let boxConstraintApplied = false;
        if (constraints) {
            let paramsCopy = null;
            paramStructure.forEach((dsStruct, dsIdx) => {
                if (!constraints[dsIdx]) return;
                dsStruct.forEach((pLen, pIdx) => {
                    if (!constraints[dsIdx][pIdx]) return;
                    for (let vIdx = 0; vIdx < pLen; vIdx++) {
                        const constraint = constraints[dsIdx]?.[pIdx]?.[vIdx];
                        if (constraint) {
                            if (!paramsCopy) paramsCopy = JSON.parse(JSON.stringify(reconstructedParams));
                            let value = paramsCopy[dsIdx][pIdx][vIdx]; const originalValue = value;
                            if (constraint.min !== undefined && value < constraint.min) value = constraint.min;
                            if (constraint.max !== undefined && value > constraint.max) value = constraint.max;
                            if (value !== originalValue) {
                                paramsCopy[dsIdx][pIdx][vIdx] = value; boxConstraintApplied = true;
                                const isActive = activeParamInfo.some(info => info.originalCoord[0][0] === dsIdx && info.originalCoord[0][1] === pIdx && info.originalCoord[1] === vIdx);
                                if (isActive) changedActive = true;
                            }
                        }
                    }
                });
            });
            if (paramsCopy) constrainedParams = paramsCopy;
        }
        if (typeof constraintFunction === 'function') {
            try {
                const paramsForCustomFn = boxConstraintApplied ? constrainedParams : reconstructedParams;
                const paramsBeforeCustom = boxConstraintApplied ? null : JSON.stringify(paramsForCustomFn);
                constrainedParams = constraintFunction(paramsForCustomFn);
                if (!Array.isArray(constrainedParams) || constrainedParams.length !== paramStructure.length) { throw new Error("Constraint function did not return a valid parameter structure."); }
                const paramsAfterCustom = JSON.stringify(constrainedParams);
                if (!boxConstraintApplied && paramsAfterCustom !== paramsBeforeCustom) { changedActive = true; logFn("Custom constraint function modified parameters.", 'debug'); }
                else if (boxConstraintApplied && paramsAfterCustom !== JSON.stringify(paramsForCustomFn)) { changedActive = true; logFn("Custom constraint function modified parameters after box constraints.", 'debug'); }
            } catch (e) { logFn(`Error executing custom constraint function: ${e.message}`, 'error'); constrainedParams = boxConstraintApplied ? JSON.parse(JSON.stringify(reconstructedParams)) : reconstructedParams; }
        }
        return { constrainedParams, changedActive };
    }

    /**
     * Calculates the final weighted residuals for each dataset after fitting.
     * @private
     */
    function calculateFinalResiduals(data, modelFunction, finalReconstructedParams, onLog) {
        if (!data || !data.x || !data.y || !data.ye || !modelFunction || !finalReconstructedParams) { onLog("Missing data/models/params for final residual calculation.", 'error'); return null; }
        const residualsPerSeries = []; let errorOccurred = false;
        data.x.forEach((xDataset, dsIdx) => {
            const yDataset = data.y[dsIdx]; const yeDataset = data.ye[dsIdx]; const models = modelFunction[dsIdx]; const paramsForDs = finalReconstructedParams[dsIdx]; const currentResiduals = [];
            if (!yDataset || !yeDataset || !models || !paramsForDs || xDataset.length !== yDataset.length || xDataset.length !== yeDataset.length) { onLog(`Inconsistent data/model/params for dataset ${dsIdx} in final residual calculation. Skipping.`, 'warn'); residualsPerSeries.push([]); return; }
            for (let ptIdx = 0; ptIdx < xDataset.length; ptIdx++) {
                const xPoint = xDataset[ptIdx]; const yPoint = yDataset[ptIdx]; const yePoint = yeDataset[ptIdx];
                if (yePoint === 0 || !isFinite(yPoint) || !isFinite(yePoint)) { onLog(`Invalid data/error at point ${ptIdx} in dataset ${dsIdx} for final residuals. Storing NaN.`, 'warn'); currentResiduals.push(NaN); continue; }
                let combinedModelValue = 0; let modelEvalError = false;
                try { models.forEach((modelFunc, paramIdx) => { if (!paramsForDs[paramIdx]) throw new Error(`Missing params for model ${paramIdx}.`); const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]); if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} returned invalid result.`); combinedModelValue += componentModelResult[0]; }); }
                catch (error) { onLog(`Error evaluating model for final residual at ds ${dsIdx}, pt ${ptIdx}: ${error.message}`, 'error'); modelEvalError = true; errorOccurred = true; }
                if (modelEvalError || !isFinite(combinedModelValue)) { currentResiduals.push(NaN); } else { currentResiduals.push((yPoint - combinedModelValue) / yePoint); }
            } residualsPerSeries.push(currentResiduals);
        });
        return errorOccurred ? null : residualsPerSeries;
    }

    /**
     * Calculates the fitted model curve over the range of the input data.
     * @private
     */
    function calculateFittedModelCurves(data, modelFunction, finalReconstructedParams, numPoints, onLog) {
         if (!data || !data.x || !modelFunction || !finalReconstructedParams || numPoints <= 1) { onLog("Missing data/models/params or invalid numPoints for fitted curve calculation.", 'error'); return null; }
        const fittedCurves = []; let errorOccurred = false;
         data.x.forEach((xDataset, dsIdx) => {
            const models = modelFunction[dsIdx]; const paramsForDs = finalReconstructedParams[dsIdx]; const currentCurve = { x: [], y: [] };
            if (!models || !paramsForDs || xDataset.length === 0) { onLog(`Inconsistent data/model/params for dataset ${dsIdx} in fitted curve calculation. Skipping.`, 'warn'); fittedCurves.push(currentCurve); return; }
            let xMin = xDataset[0]; let xMax = xDataset[0]; for(let i=1; i<xDataset.length; ++i) { if (xDataset[i] < xMin) xMin = xDataset[i]; if (xDataset[i] > xMax) xMax = xDataset[i]; }
            if (!isFinite(xMin) || !isFinite(xMax) || xMax < xMin) { onLog(`Invalid X range [${xMin}, ${xMax}] for dataset ${dsIdx} in fitted curve calculation. Skipping.`, 'warn'); fittedCurves.push(currentCurve); return; }
            const dx = (xMax === xMin) ? 0 : (xMax - xMin) / (numPoints - 1);
            for (let i = 0; i < numPoints; i++) {
                const xCalc = (numPoints === 1 || dx === 0) ? xMin : xMin + i * dx; let yCalc = 0; let modelEvalError = false;
                try { models.forEach((modelFunc, paramIdx) => { if (!paramsForDs[paramIdx]) throw new Error(`Missing params for model ${paramIdx}.`); const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xCalc]); if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} returned invalid result.`); yCalc += componentModelResult[0]; }); }
                catch(error) { onLog(`Error evaluating model for fitted curve at ds ${dsIdx}, x=${xCalc}: ${error.message}`, 'error'); modelEvalError = true; errorOccurred = true; }
                currentCurve.x.push(xCalc); currentCurve.y.push(modelEvalError || !isFinite(yCalc) ? NaN : yCalc);
            } fittedCurves.push(currentCurve);
         });
         return errorOccurred ? null : fittedCurves;
    }

    /**
     * Helper function to get details needed for error reconstruction.
     * Uses NEW linkMap structure logic. (v1.4.3 style)
     * @private
     */
    function setupParameterMappingDetails(initialParameters, linkMapInput, fixMapInput) { // linkMapInput is NEW format here
        // --- 1. Flatten Parameters and Create Coordinate Mapping ---
        const paramStructure = []; const paramCoordinates = []; let currentFlatIndex = 0;
        const fixMap = fixMapInput ? JSON.parse(JSON.stringify(fixMapInput)) : [];
        const linkMap = linkMapInput ? JSON.parse(JSON.stringify(linkMapInput)) : null;
        initialParameters.forEach((dsParams, dsIdx) => {
            paramStructure.push([]); if (!fixMap[dsIdx]) fixMap[dsIdx] = [];
            dsParams.forEach((pArray, pIdx) => {
                paramStructure[dsIdx].push(pArray.length); if (!fixMap[dsIdx][pIdx]) fixMap[dsIdx][pIdx] = new Array(pArray.length).fill(false);
                pArray.forEach((pValue, vIdx) => {
                    paramCoordinates.push([[dsIdx, pIdx], vIdx]); // Store coordinate
                    if (fixMap[dsIdx][pIdx].length <= vIdx) fixMap[dsIdx][pIdx][vIdx] = false;
                    currentFlatIndex++;
                });
            });
        });
        const nTotalParams = currentFlatIndex;
        const masterMap = new Array(nTotalParams).fill(-1); // Ensure initialized
        const isFixed = new Array(nTotalParams).fill(false); // Ensure initialized

        // --- 2. Apply fixMap ---
        paramCoordinates.forEach((coord, flatIdx) => { const [[dsIdx, paramIdx], valIdx] = coord; if (fixMap[dsIdx]?.[paramIdx]?.[valIdx] === true) isFixed[flatIdx] = true; });

        // --- 3. Process linkMap (NEW STRUCTURE LOGIC - v1.4.3 style) ---
        const linkGroupsById = {};
        if (linkMap) {
            // Iterate through linkMap structure to find groups
            linkMap.forEach((dsLinkMap, dsLinkIdx) => {
                if (!dsLinkMap || dsLinkIdx >= initialParameters.length) return;
                dsLinkMap.forEach((modelLinkMap, modelLinkIdx) => {
                    if (!modelLinkMap || modelLinkIdx >= initialParameters[dsLinkIdx].length) return;
                    modelLinkMap.forEach((groupId, paramLinkIdx) => {
                        if (paramLinkIdx >= initialParameters[dsLinkIdx][modelLinkIdx].length) return;
                        if (groupId !== null && groupId !== undefined && groupId !== '') {
                            // Find flatIndex by matching coordinates
                            const targetCoord = [dsLinkIdx, modelLinkIdx, paramLinkIdx];
                            const flatIdx = paramCoordinates.findIndex(pc =>
                                pc[0][0] === targetCoord[0] &&
                                pc[0][1] === targetCoord[1] &&
                                pc[1] === targetCoord[2]
                            );
                            if (flatIdx !== -1) {
                                if (!linkGroupsById[groupId]) linkGroupsById[groupId] = [];
                                linkGroupsById[groupId].push(flatIdx);
                            }
                        }
                    });
                });
            });

            // Process collected groups
            Object.values(linkGroupsById).forEach(flatIndices => {
                 if (flatIndices.length < 2) return;
                let masterFlatIndex = -1;
                let allInitiallyFixed = true;
                for (const flatIdx of flatIndices) {
                     if (flatIdx < 0 || flatIdx >= nTotalParams) continue;
                    if (!isFixed[flatIdx]) {
                        allInitiallyFixed = false;
                        if (masterFlatIndex === -1) { masterFlatIndex = flatIdx; }
                    }
                 }
                if (allInitiallyFixed) {
                    masterFlatIndex = flatIndices[0];
                     if (masterFlatIndex < 0 || masterFlatIndex >= nTotalParams) return;
                    if (!isFixed[masterFlatIndex]) { isFixed[masterFlatIndex] = true; }
                    flatIndices.forEach(flatIdx => {
                         if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) { masterMap[flatIdx] = masterFlatIndex; }
                    });
                } else if (masterFlatIndex !== -1) {
                    flatIndices.forEach(flatIdx => {
                         if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) {
                            if (isFixed[flatIdx]) isFixed[flatIdx] = false;
                            masterMap[flatIdx] = masterFlatIndex;
                        }
                    });
                }
            });
        }
        return { isFixed, masterMap, paramCoordinates, nTotalParams, paramStructure };
    } // <-- End of setupParameterMappingDetails (New Logic v1.4.3 style)


    // ============================================================================
    // Main Global Fit Function (Exported)
    // ============================================================================

    /**
     * Performs global curve fitting for multiple datasets using the Levenberg-Marquardt algorithm.
     * Allows linking and fixing parameters across datasets and uses composite models.
     * Includes calculation of goodness-of-fit statistics.
     *
     * @param {object} data - Contains the experimental data.
     *   @param {number[][]} data.x - Array of arrays of independent variable values.
     *   @param {number[][]} data.y - Array of arrays of dependent variable values.
     *   @param {number[][]} data.ye - Array of arrays of error values for y.
     * @param {Function[][]} modelFunction - Array of arrays of model functions.
     * @param {number[][][]} initialParameters - Nested array of initial parameter guesses.
     * @param {object} [options={}] - Optional configuration for the fitting process.
     *   @param {number} [options.maxIterations=100]
     *   @param {number} [options.errorTolerance=1e-6]
     *   @param {number} [options.gradientTolerance=1e-6]
     *   @param {(number|string|null)[][][]} [options.linkMap=null] - NEW: Nested array matching initialParameters. Non-null values indicate link group ID.
     *   @param {boolean[][][]} [options.fixMap=null] - Nested array matching initialParameters. True indicates fixed.
     *   @param {object[][][]} [options.constraints=null] - Box constraints {min, max}, nested array matching initialParameters.
     *   @param {Function | null} [options.constraintFunction=null] - Custom function: `(params) => modifiedParams`. Applied after box constraints.
     *   @param {string} [options.logLevel='info'] - Control logging verbosity ('none', 'error', 'warn', 'info', 'debug').
     *   @param {Function} [options.onLog=()=>{}] - Callback for logs: `(message, level) => {}`.
     *   @param {Function} [options.onProgress=()=>{}] - Callback for progress: `(progressData) => {}`, where `progressData = { iteration, chiSquared, lambda, activeParameters }`.
     *   @param {number|null} [options.robustCostFunction=null] - null, 1 (L1), or 2 (Lorentzian).
     *   @param {number} [options.lambdaInitial=1e-3]
     *   @param {number} [options.lambdaIncreaseFactor=10]
     *   @param {number} [options.lambdaDecreaseFactor=10]
     *   @param {number} [options.epsilon=1e-8] - Step size for numerical differentiation.
     *   @param {boolean | {numPoints: number}} [options.calculateFittedModel=false] - If true or object, calculate fitted curve.
     *   @param {number} [options.covarianceLambda=1e-9] - Regularization factor added to Hessian diagonal for covariance calculation.
     * @returns {object} - Fitting results including statistics.
     *   @property {number[]} p_active - Final values of active parameters.
     *   @property {number[][][]} p_reconstructed - Final values of all parameters (including fixed/linked) in original nested structure.
     *   @property {(number|null)[][][]} finalParamErrors - Estimated standard errors for all parameters (NaN if calculation failed, 0 for fixed).
     *   @property {number} chiSquared - Final chi-squared value.
     *   @property {number[][] | null} covarianceMatrix - Calculated covariance matrix for active parameters (potentially regularized). Null if K=0 or inversion failed.
     *   @property {number[]} parameterErrors - Estimated standard errors for active parameters (NaN if calculation failed).
     *   @property {number} iterations - Number of iterations performed.
     *   @property {boolean} converged - True if the fit converged based on tolerances.
     *   @property {string[]} activeParamLabels - Labels for the active parameters.
     *   @property {string | null} error - Error message if fit failed catastrophically.
     *   @property {number} totalPoints - Total number of data points used (N).
     *   @property {number} degreesOfFreedom - Degrees of freedom (N-K).
     *   @property {number} reducedChiSquared - Chi-squared divided by degrees of freedom.
     *   @property {number} aic - Akaike Information Criterion.
     *   @property {number} aicc - Corrected Akaike Information Criterion.
     *   @property {number} bic - Bayesian Information Criterion.
     *   @property {number[][] | null} residualsPerSeries - Weighted residuals ((y-ymodel)/ye) for each dataset.
     *   @property {{x: number[], y: number[]}[] | null} fittedModelCurves - Calculated fitted model curves for each dataset.
     */
    function lmFitGlobal(data, modelFunction, initialParameters, options = {}) {
        // --- Options Processing ---
        const maxIterations = options.maxIterations ?? 100;
        const errorTolerance = options.errorTolerance ?? 1e-6;
        const gradientTolerance = options.gradientTolerance ?? 1e-6;
        const linkMapInput = options.linkMap ?? null; // NEW format
        const fixMapInput = options.fixMap ?? null;
        const constraints = options.constraints ?? null;
        const constraintFunction = options.constraintFunction ?? null;
        const logLevelStr = options.logLevel ?? 'info';
        const logLevel = LOG_LEVELS[logLevelStr.toLowerCase()] ?? LOG_LEVELS.info;
        const onLog = options.onLog && typeof options.onLog === 'function' ? options.onLog : () => {};
        const onProgress = options.onProgress && typeof options.onProgress === 'function' ? options.onProgress : () => {};
        const robustCostFunction = options.robustCostFunction ?? null;
        let lambda = options.lambdaInitial ?? 1e-3;
        const lambdaIncreaseFactor = options.lambdaIncreaseFactor ?? 10;
        const lambdaDecreaseFactor = options.lambdaDecreaseFactor ?? 10;
        const epsilon = options.epsilon ?? 1e-8;
        const calculateFittedOpt = options.calculateFittedModel ?? false;
        const numPointsForCurve = (typeof calculateFittedOpt === 'object' && calculateFittedOpt.numPoints > 1)
                                  ? calculateFittedOpt.numPoints
                                  : 300;
        const shouldCalculateFitted = calculateFittedOpt === true || (typeof calculateFittedOpt === 'object');
        // *** Regularization parameter for covariance calculation ***
        const covarianceLambda = options.covarianceLambda ?? 1e-9;

        // Internal logging helpers respecting logLevel
        const logFn = (message, level) => { const messageLevel = LOG_LEVELS[level] ?? LOG_LEVELS.info; if (logLevel >= messageLevel) { onLog(message, level); } };
        const logInfo = (message) => logFn(message, 'info');
        const logWarn = (message) => logFn(message, 'warn');
        const logError = (message) => logFn(message, 'error');
        const logDebug = (message) => logFn(message, 'debug');

        logInfo("Starting lmFitGlobal (Using New linkMap Logic v1.2.5)..."); // Update version marker

        // --- Calculate Total Data Points (N) ---
        let totalPoints = 0;
        if (data && data.x) { data.x.forEach(xDataset => { if (Array.isArray(xDataset)) totalPoints += xDataset.length; else logWarn("Non-array in data.x."); }); }
        logInfo(`Total data points (N): ${totalPoints}`);
        const baseErrorReturn = { p_active: [], p_reconstructed: initialParameters, finalParamErrors: null, chiSquared: NaN, covarianceMatrix: null, parameterErrors: null, iterations: 0, converged: false, activeParamLabels: [], totalPoints: totalPoints, degreesOfFreedom: NaN, reducedChiSquared: NaN, aic: NaN, aicc: NaN, bic: NaN, residualsPerSeries: null, fittedModelCurves: null };
        if (totalPoints === 0) { const errMsg = "No data points."; logError(errMsg); return { ...baseErrorReturn, error: errMsg }; }

        // --- Parameter Mapping Setup (K = totalActiveParams) ---
        let setupResults;
         try {
             // *** Pass linkMapInput (new format) to the updated setup function ***
             setupResults = setupParameterMapping(initialParameters, linkMapInput, fixMapInput, logDebug);
         }
         catch (error) { const errMsg = `Parameter setup failed: ${error.message}`; logError(errMsg); return { ...baseErrorReturn, error: errMsg }; }

        const { activeInitialParams, reconstructParams, activeParamInfo, totalActiveParams, paramStructure, activeParamLabels } = setupResults;
        const K = totalActiveParams;
        // Get mapping details needed for error reconstruction (using new logic)
        const { isFixed, masterMap, paramCoordinates, nTotalParams } = setupParameterMappingDetails(initialParameters, linkMapInput, fixMapInput);

        // --- Handle Case: No Active Parameters ---
        if (K === 0) {
            // ... (same as previous versions) ...
             logWarn("No active parameters to fit. Calculating initial stats.");
            let initialChiSq = NaN; let dof = totalPoints; let redChiSq = NaN, aic = NaN, aicc = NaN, bic = NaN;
            let finalResiduals = null; let fittedCurves = null; let finalErrors = null;
            const initialReconstructed = reconstructParams([]);
            try {
                initialChiSq = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, [], robustCostFunction, paramStructure, logFn);
                finalResiduals = calculateFinalResiduals(data, modelFunction, initialReconstructed, logFn);
                if (shouldCalculateFitted) { fittedCurves = calculateFittedModelCurves(data, modelFunction, initialReconstructed, numPointsForCurve, logFn); }
                finalErrors = []; let currentFlatIdx_err = 0;
                paramStructure.forEach((dsStruct, dsIdx) => { finalErrors[dsIdx] = []; dsStruct.forEach((pLen, pIdx) => { finalErrors[dsIdx][pIdx] = []; for (let vIdx = 0; vIdx < pLen; vIdx++) { finalErrors[dsIdx][pIdx][vIdx] = 0; currentFlatIdx_err++; } }); });
                if (isFinite(initialChiSq) && dof > 0) { redChiSq = initialChiSq / dof; aic = initialChiSq; bic = initialChiSq; if (totalPoints > 1) aicc = aic; else aicc = Infinity; }
                 else if (isFinite(initialChiSq)) { redChiSq = Infinity; aic = initialChiSq; bic = initialChiSq; aicc = Infinity; logWarn("Degrees of freedom is zero or negative. Reduced ChiSq and AICc are Infinity."); }
            } catch(e) { logError(`Error calculating initial ChiSq/Residuals: ${e.message}`); }
            return { p_active: [], p_reconstructed: initialReconstructed, finalParamErrors: finalErrors, chiSquared: initialChiSq, covarianceMatrix: [], parameterErrors: [], iterations: 0, converged: true, activeParamLabels: [], totalPoints: totalPoints, degreesOfFreedom: dof, reducedChiSquared: redChiSq, aic: aic, aicc: aicc, bic: bic, error: null, residualsPerSeries: finalResiduals, fittedModelCurves: fittedCurves };
        }

        // --- Initial Calculations ---
        let activeParameters = [...activeInitialParams];
        let chiSquared = NaN;
        try { chiSquared = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, activeParameters, robustCostFunction, paramStructure, logFn); }
        catch (error) { const errMsg = `Initial Chi-Squared calculation failed: ${error.message}`; logError(errMsg); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), error: errMsg }; }

        let converged = false; let covarianceMatrix = null; let parameterErrors = []; let iterationsPerformed = 0; let finalHessian = null;

        logInfo(`Total active parameters (K): ${K}`); logInfo(`Active Parameter Labels: ${activeParamLabels.join(', ')}`); logInfo(`Initial Active Parameters: ${activeParameters.map(p=>p.toExponential(3)).join(', ')}`); logInfo(`Initial Chi-Squared: ${chiSquared}`);

        if (!isFinite(chiSquared)) { const errMsg = "Initial Chi-Squared is not finite."; logError(errMsg); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, error: errMsg }; }

        // --- LM Iteration Loop ---
        // ... (Iteration loop is identical to previous versions) ...
        let iteration;
        for (iteration = 0; iteration < maxIterations; iteration++) {
            iterationsPerformed = iteration + 1; logInfo(`--- Iteration ${iterationsPerformed} (Lambda: ${lambda.toExponential(3)}) ---`); logDebug(`Iter ${iterationsPerformed} - Current Active Params: ${activeParameters.map(p=>p.toExponential(5)).join(', ')}`);
            let jacobian, residuals; try { ({ jacobian, residuals } = calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParams, activeParameters, activeParamInfo, epsilon, paramStructure, logFn)); } catch (error) { const errMsg = `Jacobian failed: ${error.message}`; logError(`Error Jacobian/Resid iter ${iterationsPerformed}: ${errMsg}`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            let gradient, currentHessian; try { gradient = calculateGradient(jacobian, residuals); currentHessian = calculateHessian(jacobian); } catch (error) { const errMsg = `Grad/Hess failed: ${error.message}`; logError(`Error Grad/Hess iter ${iterationsPerformed}: ${errMsg}`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            if (gradient.some(g => !isFinite(g)) || currentHessian.some(row => row.some(h => !isFinite(h)))) { const errMsg = "Non-finite grad/hess."; logError(`Non-finite grad/hess iter ${iterationsPerformed}.`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            const maxGradient = Math.max(...gradient.map(Math.abs)); if (maxGradient < gradientTolerance) { logInfo(`Converged (grad tol ${gradientTolerance}). Max Grad: ${maxGradient.toExponential(3)}`); converged = true; break; }
            logInfo(`Gradient: ${gradient.map(g => g.toExponential(3)).join(', ')}`);
            let parameterUpdates; let solveSuccess = false; let attempt = 0; const maxSolveAttempts = 5; let currentLambda = lambda;
            while (!solveSuccess && attempt < maxSolveAttempts) {
                const dampedHessian = currentHessian.map((row, i) => row.map((value, j) => (i === j ? value + currentLambda : value)));
                try { const negativeGradient = gradient.map(g => -g); parameterUpdates = solveLinearSystem(dampedHessian, negativeGradient); if (parameterUpdates.some(up => !isFinite(up))) throw new Error("NaN/Inf in updates."); solveSuccess = true; }
                catch (error) { attempt++; logWarn(`Solve failed (Att ${attempt}/${maxSolveAttempts}, Iter ${iterationsPerformed}): ${error.message}. Inc lambda.`); currentLambda = Math.min(currentLambda * lambdaIncreaseFactor * (attempt > 1 ? lambdaIncreaseFactor : 1) , 1e10); logInfo(`Attempting solve with Lambda: ${currentLambda.toExponential(3)}`); if (attempt >= maxSolveAttempts) { const errMsg = "Failed solve."; logError(`Failed solve after ${maxSolveAttempts} attempts.`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; } }
            }
            if (!solveSuccess) continue; lambda = currentLambda; logInfo(`Parameter Updates: ${parameterUpdates.map(pu => pu.toExponential(3)).join(', ')}`);
            const proposedActiveParams = activeParameters.map((p, i) => p + parameterUpdates[i]);
            let proposedReconstructed = reconstructParams(proposedActiveParams);
            const { constrainedParams, changedActive } = applyConstraintsGlobal( proposedReconstructed, constraints, activeParamInfo, paramStructure, constraintFunction, logFn );
            proposedReconstructed = constrainedParams;
            let finalProposedActiveParams = [...proposedActiveParams];
            if (changedActive) { const tempActive = []; activeParamInfo.forEach(info => { const [[ds, p], v] = info.originalCoord; tempActive.push(proposedReconstructed[ds][p][v]); }); finalProposedActiveParams = tempActive; logInfo("Constraints modified parameters, re-extracted active values."); }
            let newChiSquared = NaN; try { newChiSquared = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, finalProposedActiveParams, robustCostFunction, paramStructure, logFn); } catch (error) { logError(`Error ChiSq proposed step iter ${iterationsPerformed}: ${error.message}`); newChiSquared = Infinity; }
            logInfo(`New Chi-Squared: ${newChiSquared}`);
            if (isFinite(newChiSquared) && newChiSquared < chiSquared) {
                const chiSquaredChange = chiSquared - newChiSquared; activeParameters = finalProposedActiveParams; chiSquared = newChiSquared; lambda = Math.max(lambda / lambdaDecreaseFactor, 1e-12);
                logInfo(`Accepted. ChiSq decreased by ${chiSquaredChange.toExponential(3)}. Lambda decreased to: ${lambda.toExponential(3)}`);
                try { onProgress({ iteration: iterationsPerformed, chiSquared: chiSquared, lambda: lambda, activeParameters: [...activeParameters] }); } catch (e) { logWarn(`Error in onProgress callback: ${e.message}`); }
                if (chiSquaredChange < errorTolerance) { logInfo(`Converged (chiSq tol ${errorTolerance}).`); converged = true; break; }
            } else { lambda = Math.min(lambda * lambdaIncreaseFactor, 1e10); logInfo(`Rejected (ChiSq ${isNaN(newChiSquared) ? 'NaN' : 'increased/stagnant'}). Lambda increased to: ${lambda.toExponential(3)}`); if (lambda >= 1e10) logWarn("Lambda reached maximum limit."); }
        }

        // --- Post-Loop Processing & Statistics ---
        if (!converged && iteration === maxIterations) { logWarn(`lmFitGlobal did not converge within ${maxIterations} iterations.`); }
        logInfo("Recalculating final Jacobian/Hessian for covariance...");
        try { const { jacobian: finalJacobian } = calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParams, activeParameters, activeParamInfo, epsilon, paramStructure, logFn); finalHessian = calculateHessian(finalJacobian); }
        catch (error) { logError(`Failed to recalculate final Hessian: ${error.message}`); finalHessian = null; }

        // --- Calculate Statistics & Parameter Errors ---
        let degreesOfFreedom = NaN, reducedChiSquared = NaN, aic = NaN, aicc = NaN, bic = NaN;
        if (isFinite(chiSquared) && totalPoints > 0) {
            degreesOfFreedom = totalPoints - K;
            if (degreesOfFreedom > 0) {
                reducedChiSquared = chiSquared / degreesOfFreedom; aic = chiSquared + 2 * K; const aiccCorrectionDenom = totalPoints - K - 1;
                if (aiccCorrectionDenom > 0) { aicc = aic + (2 * K * (K + 1)) / aiccCorrectionDenom; } else { aicc = Infinity; logWarn("AICc denominator (N-K-1) is zero or negative. AICc set to Infinity."); }
                bic = chiSquared + K * Math.log(totalPoints);
            } else { reducedChiSquared = Infinity; aic = chiSquared + 2 * K; bic = chiSquared + K * Math.log(totalPoints); aicc = Infinity; logWarn(`Degrees of freedom (${degreesOfFreedom}) is zero or negative. Reduced Chi-Squared is Infinity/undefined. Parameter errors may be unreliable or NaN.`); }
        } else { logWarn("Final Chi-Squared is not finite. Cannot calculate statistics reliably."); degreesOfFreedom = totalPoints - K; reducedChiSquared = NaN; aic = NaN; aicc = NaN; bic = NaN; }

        // Initialize error outputs
        parameterErrors = new Array(K).fill(NaN);
        covarianceMatrix = null; // Initialize as null

        // Attempt to calculate covariance and errors only if Hessian is available and K > 0
        if (finalHessian && K > 0) {
            try {
                // *** Apply Regularization before inversion ***
                const regularizedHessian = finalHessian.map((row, i) =>
                    row.map((value, j) => (i === j ? value + covarianceLambda : value))
                );
                logDebug(`Applying regularization (lambda=${covarianceLambda}) for covariance matrix inversion.`);
                covarianceMatrix = invertMatrixUsingSVD(regularizedHessian); // Invert the regularized matrix

                const scaleFactor = (reducedChiSquared && isFinite(reducedChiSquared) && reducedChiSquared > 0) ? reducedChiSquared : 1.0;
                if (scaleFactor === 1.0 && degreesOfFreedom > 0 && isFinite(chiSquared)) { logWarn("Reduced Chi-Squared is invalid or not positive. Using scale factor 1.0 for parameter errors, which might underestimate errors if fit is poor."); }
                else if (scaleFactor === 1.0 && degreesOfFreedom <= 0) { logInfo("Using scale factor 1.0 for parameter errors due to non-positive degrees of freedom."); }

                // Calculate parameter errors from diagonal of covariance matrix
                parameterErrors = covarianceMatrix.map((row, i) => {
                    if (i >= row.length) { logError(`Error accessing covariance matrix diagonal at index ${i}. Matrix might be malformed.`); return NaN; }
                    const variance = row[i];
                    const scaledVariance = variance * scaleFactor;

                    // *** Use Math.abs() for standard error calculation ***
                    let error = NaN;
                    if (isFinite(scaledVariance)) {
                        if (scaledVariance < 0) {
                            logWarn(`Negative variance (${scaledVariance.toExponential(3)}) encountered for active param ${i}. Returning sqrt(abs(variance)). Error estimate might be unreliable.`);
                            error = Math.sqrt(Math.abs(scaledVariance));
                        } else {
                            error = Math.sqrt(scaledVariance);
                        }
                    }
                    // Log if still NaN for other reasons (e.g., variance was NaN/Infinity)
                    if (isNaN(error) && isFinite(scaledVariance)) {
                         logDebug(`NaN error calc for param ${i}: variance=${variance}, scaleFactor=${scaleFactor}, scaledVariance=${scaledVariance}`);
                    }
                    return error;
                });

                // Update general warning
                if (parameterErrors.some(isNaN)) {
                    logWarn("NaN encountered in parameter errors (potentially due to non-finite variance/covariance). Check fit quality, model, initial parameters, and data.");
                }

            } catch (error) {
                logError(`Failed to calculate covariance matrix/parameter errors: ${error.message}`);
                parameterErrors = new Array(K).fill(NaN);
                covarianceMatrix = new Array(K).fill(0).map(() => new Array(K).fill(NaN)); // Set to NaN matrix on error
            }
        } else {
            if (K > 0 && !finalHessian) { logWarn("Could not calculate covariance matrix (no valid final Hessian?). Parameter errors will be NaN."); }
            else if (K === 0) { logInfo("No active parameters (K=0). Parameter errors are not applicable."); }
            covarianceMatrix = new Array(K).fill(0).map(() => new Array(K).fill(NaN)); // Ensure it's NaN matrix if not calculated
        }

        const finalReconstructedParams = reconstructParams(activeParameters);
        let finalParamErrors = null;
        try {
            finalParamErrors = []; let currentFlatIdx_err = 0;
            paramStructure.forEach((dsStruct, dsIdx) => { finalParamErrors[dsIdx] = []; dsStruct.forEach((pLen, pIdx) => { finalParamErrors[dsIdx][pIdx] = []; for (let vIdx = 0; vIdx < pLen; vIdx++) { let errorValue = NaN; if (isFixed[currentFlatIdx_err]) { errorValue = 0; } else if (masterMap[currentFlatIdx_err] !== -1) { const masterFlatIdx = masterMap[currentFlatIdx_err]; const masterInfo = activeParamInfo.find(info => info.flatIndex === masterFlatIdx); if (masterInfo) { if(parameterErrors && masterInfo.activeIndex < parameterErrors.length) { errorValue = parameterErrors[masterInfo.activeIndex]; } else { logWarn(`Invalid index ${masterInfo.activeIndex} for parameterErrors array (length ${parameterErrors?.length}) while processing slave ${currentFlatIdx_err}. Setting error to NaN.`); errorValue = NaN; } } else { logWarn(`Could not find active info for master parameter (flat index ${masterFlatIdx}) of slave (flat index ${currentFlatIdx_err}). Setting error to 0.`); errorValue = 0; } } else { const activeInfo = activeParamInfo.find(info => info.flatIndex === currentFlatIdx_err); if (activeInfo) { if(parameterErrors && activeInfo.activeIndex < parameterErrors.length) { errorValue = parameterErrors[activeInfo.activeIndex]; } else { logWarn(`Invalid index ${activeInfo.activeIndex} for parameterErrors array (length ${parameterErrors?.length}) while processing active param ${currentFlatIdx_err}. Setting error to NaN.`); errorValue = NaN; } } else { logWarn(`Could not find active info for supposedly active parameter at flat index ${currentFlatIdx_err}. Setting error to NaN.`); } } finalParamErrors[dsIdx][pIdx][vIdx] = errorValue; currentFlatIdx_err++; } }); });
        } catch (e) { logError(`Error constructing finalParamErrors structure: ${e.message}`); finalParamErrors = null; }

        let finalResiduals = null; try { finalResiduals = calculateFinalResiduals(data, modelFunction, finalReconstructedParams, logFn); } catch (error) { logError(`Failed to calculate final residuals: ${error.message}`); }
        let fittedCurves = null; if (shouldCalculateFitted) { logInfo(`Calculating fitted model curves with ${numPointsForCurve} points...`); try { fittedCurves = calculateFittedModelCurves(data, modelFunction, finalReconstructedParams, numPointsForCurve, logFn); } catch (error) { logError(`Failed to calculate fitted model curves: ${error.message}`); } }

        logInfo("--------------------"); logInfo("lmFitGlobal Finished."); logInfo(`Iterations Performed: ${iterationsPerformed}`); logInfo(`Total Points (N): ${totalPoints}`); logInfo(`Active Parameters (K): ${K}`); logInfo(`Degrees of Freedom (N-K): ${degreesOfFreedom}`); logInfo(`Final Active Parameters: ${activeParameters.map(p=>p.toExponential(5)).join(', ')}`); logInfo(`Final Chi-Squared: ${chiSquared}`); logInfo(`Reduced Chi-Squared: ${reducedChiSquared}`); logInfo(`AIC: ${aic}`); logInfo(`AICc: ${aicc}`); logInfo(`BIC: ${bic}`); logInfo(`Parameter Errors (Active): ${parameterErrors.map(p=>isNaN(p)?'NaN':p.toExponential(3)).join(', ')}`); logInfo(`Converged: ${converged}`); logInfo("--------------------");

        // *** Return covarianceMatrix ***
        return {
            p_active: activeParameters, p_reconstructed: finalReconstructedParams, finalParamErrors,
            chiSquared, covarianceMatrix, parameterErrors, iterations: iterationsPerformed, converged,
            activeParamLabels, error: null, totalPoints, degreesOfFreedom, reducedChiSquared,
            aic, aicc, bic, residualsPerSeries: finalResiduals, fittedModelCurves: fittedCurves
        };
    } // <-- End of lmFitGlobal function definition


    // ============================================================================
    // Helper Functions for Wrappers
    // ============================================================================
     function isSingleDataset(data) { return data && Array.isArray(data.x) && (data.x.length === 0 || !Array.isArray(data.x[0])); }
     function wrapSingleDatasetInput(data, modelFunction, initialParameters, options) {
        const wrappedData = { x: [data.x], y: [data.y], ye: [data.ye] };
        const wrappedModelFunction = [ Array.isArray(modelFunction) ? modelFunction : [modelFunction] ];
        const wrappedInitialParameters = [ initialParameters ];
        const wrappedOptions = { ...options };
        if (options.fixMap && (!Array.isArray(options.fixMap[0]) || !Array.isArray(options.fixMap[0][0]))) { wrappedOptions.fixMap = [options.fixMap]; }
        if (options.constraints && (!Array.isArray(options.constraints[0]) || !Array.isArray(options.constraints[0][0]))) { wrappedOptions.constraints = [options.constraints]; }
        // *** Wrap linkMap (NEW format) ***
        if (options.linkMap && (!Array.isArray(options.linkMap[0]) || !Array.isArray(options.linkMap[0][0]) || !Array.isArray(options.linkMap[0][0][0]))) {
             wrappedOptions.linkMap = [options.linkMap];
        }
        return { data: wrappedData, modelFunction: wrappedModelFunction, initialParameters: wrappedInitialParameters, options: wrappedOptions };
    }

    // ============================================================================
    // User-Facing Wrapper Functions
    // ============================================================================
    /**
     * Fits a single dataset using the Levenberg-Marquardt algorithm.
     * @param {object} data - {x: number[], y: number[], ye: number[]}.
     * @param {Function | Function[]} modelFunction - A single model function or an array of model functions.
     * @param {number[] | number[][]} initialParameters - Initial guesses. Nested if multiple models.
     * @param {object} [options={}] - Optional configuration (see lmFitGlobal docs).
     *                                 `fixMap`, `constraints` in single-dataset format.
     *                                 `linkMap` in new single-dataset format (e.g., `linkMap = [[null, 'id1'], ['id1', null]]`).
     * @returns {object} The fitting result object.
     */
    function lmFit(data, modelFunction, initialParameters, options = {}) {
         if (!isSingleDataset(data)) { throw new Error("lmFit requires single dataset input format."); }
         const modelFuncArray = Array.isArray(modelFunction) ? modelFunction : [modelFunction];
         const initialParamsNested = modelFuncArray.length === 1 && initialParameters.length > 0 && !Array.isArray(initialParameters[0]) ? [initialParameters] : initialParameters;
         const { data: wrappedData, modelFunction: wrappedModelFunc, initialParameters: wrappedInitialParams, options: wrappedOptions } = wrapSingleDatasetInput(data, modelFuncArray, initialParamsNested, options);
         const result = lmFitGlobal(wrappedData, wrappedModelFunc, wrappedInitialParams, wrappedOptions);
         return result;
    }

    /**
     * Fits multiple datasets independently (sequentially) using lmFitGlobal.
     * @param {object} data - {x: number[][], y: number[][], ye: number[][]}.
     * @param {Function[][]} modelFunction - Array of arrays of model functions.
     * @param {number[][][]} initialParameters - Nested array of initial parameter guesses.
     * @param {object} [options={}] - Optional configuration. `linkMap` is ignored.
     * @returns {object[]} An array of fitting result objects.
     */
    function lmFitIndependent(data, modelFunction, initialParameters, options = {}) {
        if (isSingleDataset(data)) {
            console.warn("lmFitIndependent received single dataset input. Calling lmFit instead.");
            try {
                 const modelFuncArray = Array.isArray(modelFunction) ? modelFunction : [modelFunction];
                 const initialParamsInput = (modelFuncArray.length === 1 && initialParameters.length > 0 && !Array.isArray(initialParameters[0])) ? initialParameters : (initialParameters[0] ?? []);
                 const modelFuncInput = modelFuncArray.length === 1 ? modelFuncArray[0] : (modelFuncArray[0] ?? []);
                const result = lmFit(data, modelFuncInput, initialParamsInput, options);
                return [result];
            } catch (e) { return [{ error: `Error processing single dataset input for lmFitIndependent: ${e.message}`, converged: false }]; }
        }
        const numDatasets = data.x?.length ?? 0; if (numDatasets === 0) return [];
        if (modelFunction?.length !== numDatasets || initialParameters?.length !== numDatasets || data.y?.length !== numDatasets || data.ye?.length !== numDatasets) { throw new Error("Input array lengths must match for lmFitIndependent."); }
        const allResults = []; const rootOnLog = options.onLog || (() => {}); const rootOnProgress = options.onProgress || (() => {}); const rootLogLevel = LOG_LEVELS[options.logLevel?.toLowerCase() ?? 'info'] ?? LOG_LEVELS.info;
        for (let i = 0; i < numDatasets; i++) {
            const datasetIndex = i;
            const logFnLoop = (message, level) => { const messageLevel = LOG_LEVELS[level] ?? LOG_LEVELS.info; if (rootLogLevel >= messageLevel) { rootOnLog(message, level, datasetIndex); } };
            logFnLoop(`--- Starting Independent Fit for Dataset ${datasetIndex} ---`, 'info');
            const singleData = { x: data.x[i], y: data.y[i], ye: data.ye[i] }; const singleModelFunc = modelFunction[i]; const singleInitialParams = initialParameters[i];
            const singleOptions = { ...options };
            if (options.fixMap && Array.isArray(options.fixMap[i])) { singleOptions.fixMap = options.fixMap[i]; } else { delete singleOptions.fixMap; }
            if (options.constraints && Array.isArray(options.constraints[i])) { singleOptions.constraints = options.constraints[i]; } else { delete singleOptions.constraints; }
            delete singleOptions.linkMap; delete singleOptions.onLog; delete singleOptions.onProgress;
            singleOptions.onLog = (message, level) => { rootOnLog(message, level, datasetIndex); }; singleOptions.onProgress = (progressData) => { rootOnProgress(progressData, datasetIndex); };
            const { data: wrappedData, modelFunction: wrappedModelFunc, initialParameters: wrappedInitialParams, options: wrappedOptions } = wrapSingleDatasetInput(singleData, singleModelFunc, singleInitialParams, singleOptions);
            try { const result = lmFitGlobal(wrappedData, wrappedModelFunc, wrappedInitialParams, wrappedOptions); allResults.push(result); }
            catch (error) { logFnLoop(`Error fitting dataset ${datasetIndex}: ${error.message}`, 'error'); allResults.push({ error: `Fit failed: ${error.message}`, converged: false, p_active: [], p_reconstructed: singleInitialParams, finalParamErrors: null, chiSquared: NaN, covarianceMatrix: null, parameterErrors: [], iterations: 0, activeParamLabels: [], totalPoints: singleData.x?.length ?? 0, degreesOfFreedom: NaN, reducedChiSquared: NaN, aic: NaN, aicc: NaN, bic: NaN, residualsPerSeries: null, fittedModelCurves: null }); }
            logFnLoop(`--- Finished Independent Fit for Dataset ${datasetIndex} ---`, 'info');
        }
        return allResults;
    }

    // Expose public functions
    global.lmFitGlobal = lmFitGlobal;
    global.lmFit = lmFit;
    global.lmFitIndependent = lmFitIndependent;

// Establish the root object, `window` in the browser, or `global` on the server.
})(typeof window !== 'undefined' ? window : global); // <-- End of the IIFE wrapper