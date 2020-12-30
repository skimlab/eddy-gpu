#include "assert.h"
#include "definitions.cuh"
#include <stdio.h>

// This function is an implementation of Algorithm AS 147:
//   available from http://ftp.uni-bayreuth.de/math/statlib/apstat/147
//
// Also refers to https://en.wikipedia.org/wiki/Chi-square_distribution
//   for what it does.
__device__ double deviceGammds(double x, double p) {
  double a;
  double arg;
  double c;
  double e = 1.0E-09;
  double f;
  // int ifault2;
  double uflo = 1.0E-37;
  double value;
  //
  //  Check the input.
  //
  if (x <= 0.0) {
    //*ifault = 1;
    value = 0.0;
    return value;
  }

  if (p <= 0.0) {
    //*ifault = 1;
    value = 0.0;
    return value;
  }
  //
  //  LGAMMA is the natural logarithm of the gamma function.
  //
  arg = p * log(x) - lgamma(p + 1.0) - x;

  if (arg < log(uflo)) {
    value = 0.0;
    //*ifault = 2;
    return value;
  }

  f = exp(arg);

  if (f == 0.0) {
    value = 0.0;
    //*ifault = 2;
    return value;
  }

  //*ifault = 0;
  //
  //  Series begins.
  //
  c = 1.0;
  value = 1.0;
  a = p;

  for (;;) {
    a = a + 1.0;
    c = c * x / a;
    value = value + c;

    if (c <= e * value) {
      break;
    }
  }

  value = value * f;

  return value;
  /*JR*/
}

__device__ double sumrtime(const int offset, const int len, int *data, int *spc,
                           int *fr, int *dof, int idx) {

  int skipper, con = 3;
  // contigency table observed
  int tally[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  // contigency table expected
  double expected[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  skipper = blockIdx.x - offset;
  for (int k1 = 0; k1 < len; k1++) {
    if ((skipper != 0) && (k1 == skipper - 1)) {
      continue;
    }
    // place tally for each occurence in observed contingency table
    if ((data[(spc[idx] * len) + k1] == -1) &&
        (data[(fr[idx] * len) + k1] == -1)) {
      tally[0][0]++;
    } else if ((data[spc[idx] * len + k1] == -1) &&
               (data[fr[idx] * len + k1] == 0)) {
      tally[0][1]++;
    } else if ((data[spc[idx] * len + k1] == -1) &&
               (data[fr[idx] * len + k1] == 1)) {
      tally[0][2]++;
    } else if ((data[spc[idx] * len + k1] == 0) &&
               (data[fr[idx] * len + k1] == -1)) {
      tally[1][0]++;
    } else if ((data[spc[idx] * len + k1] == 0) &&
               (data[fr[idx] * len + k1] == 0)) {
      tally[1][1]++;
    } else if ((data[spc[idx] * len + k1] == 0) &&
               (data[fr[idx] * len + k1] == 1)) {
      tally[1][2]++;
    } else if ((data[spc[idx] * len + k1] == 1) &&
               (data[fr[idx] * len + k1] == -1)) {
      tally[2][0]++;
    } else if ((data[spc[idx] * len + k1] == 1) &&
               (data[fr[idx] * len + k1] == 0)) {
      tally[2][1]++;
    } else if ((data[spc[idx] * len + k1] == 1) &&
               (data[fr[idx] * len + k1] == 1)) {
      tally[2][2]++;
    }
  }

  // summation of rows and columns for chi squared table
  int ex[7] = {0, 0, 0, 0, 0, 0, 0};
  double yates = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if (c1 == 0) {
        ex[0] += tally[c][c1];
      } else if (c1 == 1) {
        ex[1] += tally[c][c1];
      } else if (c1 == 2) {
        ex[2] += tally[c][c1];
      }
    }
    for (int b = 0; b < con; b++) {
      if (b == 0) {
        ex[3] += tally[b][c];
      } else if (b == 1) {
        ex[4] += tally[b][c];
      } else if (b == 2) {
        ex[5] += tally[b][c];
      }
    }
  }

  if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5])) {
    printf("bad math!!!!!!!!");
  } else {
    ex[6] = ex[0] + ex[1] + ex[2];
    // printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d
    // %d %d %d***",
    //       idx, tally[0][0], tally[0][1], tally[0][2],
    //            tally[1][0], tally[1][1], tally[1][2],
    //            tally[2][0], tally[2][1], tally[2][2],
    //            ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
  }
  double divisor = double(ex[6]);
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      expected[c][c1] = (double(ex[c1]) * double(ex[c + 3]) / divisor);
    }
  }

  // set use of yates correction if 1 cell < 5
  int flag = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))) {
        yates = .5;
        flag = 1;
        break;
      }
    }
    if (flag)
      break;
  }

  double chiSm = 0;
  int dofn = 0;
  int dofm = 0;

  // calculating chi squared sum
  for (int ii = 0; ii < 3; ii++) {
    if (ex[ii] == 0) {
      dofm++;
    }
    if (ex[ii + 3] == 0) {
      dofn++;
    }
    for (int jj = 0; jj < 3; jj++) {
      // save calculation time if not zero
      if ((ex[jj] * ex[ii + 3]) != 0) {
        chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) /
                 expected[ii][jj];
      }
    }
  }

  dof[threadIdx.x + blockDim.x * blockIdx.x] =
      ((3 - dofm) - 1) * ((3 - dofn) - 1);

  return chiSm;
  // return tally[0][0];
}

__device__ double sumrtimeScalable(const int offset, const int len, int *data,
                                   int *spc, int *fr, int *dof, int idx,
                                   int netID, int globalIdx) {
  int skipper, con = 3;
  // contigency table observed
  int tally[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  // contigency table expected
  double expected[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // skipper = blockIdx.x - offset;
  skipper = netID - offset;
  for (int k1 = 0; k1 < len; k1++) {
    if ((skipper != 0) && (k1 == skipper - 1)) {
      continue;
    }
    // place tally for each occurence in observed contingency table
    if ((data[(spc[idx] * len) + k1] == -1) &&
        (data[(fr[idx] * len) + k1] == -1)) {
      tally[0][0]++;
    } else if ((data[spc[idx] * len + k1] == -1) &&
               (data[fr[idx] * len + k1] == 0)) {
      tally[0][1]++;
    } else if ((data[spc[idx] * len + k1] == -1) &&
               (data[fr[idx] * len + k1] == 1)) {
      tally[0][2]++;
    } else if ((data[spc[idx] * len + k1] == 0) &&
               (data[fr[idx] * len + k1] == -1)) {
      tally[1][0]++;
    } else if ((data[spc[idx] * len + k1] == 0) &&
               (data[fr[idx] * len + k1] == 0)) {
      tally[1][1]++;
    } else if ((data[spc[idx] * len + k1] == 0) &&
               (data[fr[idx] * len + k1] == 1)) {
      tally[1][2]++;
    } else if ((data[spc[idx] * len + k1] == 1) &&
               (data[fr[idx] * len + k1] == -1)) {
      tally[2][0]++;
    } else if ((data[spc[idx] * len + k1] == 1) &&
               (data[fr[idx] * len + k1] == 0)) {
      tally[2][1]++;
    } else if ((data[spc[idx] * len + k1] == 1) &&
               (data[fr[idx] * len + k1] == 1)) {
      tally[2][2]++;
    }
  }

  // summation of rows and columns for chi squared table
  int ex[7] = {0, 0, 0, 0, 0, 0, 0};
  double yates = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if (c1 == 0) {
        ex[0] += tally[c][c1];
      } else if (c1 == 1) {
        ex[1] += tally[c][c1];
      } else if (c1 == 2) {
        ex[2] += tally[c][c1];
      }
    }
    for (int b = 0; b < con; b++) {
      if (b == 0) {
        ex[3] += tally[b][c];
      } else if (b == 1) {
        ex[4] += tally[b][c];
      } else if (b == 2) {
        ex[5] += tally[b][c];
      }
    }
  }

  if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5])) {
    printf("bad math!!!!!!!!");
  } else {
    ex[6] = ex[0] + ex[1] + ex[2];
    // printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d
    // %d %d %d***",
    //                idx,
    //                tally[0][0], tally[0][1], tally[0][2],
    //                tally[1][0], tally[1][1], tally[1][2],
    //                tally[2][0], tally[2][1], tally[2][2],
    //                ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
  }

  double divisor = double(ex[6]);
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      expected[c][c1] = (double(ex[c1]) * double(ex[c + 3]) / divisor);
    }
  }

  // set use of yates correction if 1 cell < 5
  int flag = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))) {
        yates = .5;
        flag = 1;
        break;
      }
    }
    if (flag)
      break;
  }

  double chiSm = 0;
  int dofn = 0;
  int dofm = 0;

  // calculating chi squared sum
  for (int ii = 0; ii < 3; ii++) {
    if (ex[ii] == 0) {
      dofm++;
    }
    if (ex[ii + 3] == 0) {
      dofn++;
    }
    for (int jj = 0; jj < 3; jj++) {
      // save calculation time if not zero
      if ((ex[jj] * ex[ii + 3]) != 0) {
        chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) /
                 expected[ii][jj];
      }
    }
  }

  //
  // dof[threadIdx.x + blockDim.x*netID] = ((3 - dofm) - 1)*((3 - dofn) - 1);
  dof[globalIdx] = ((3 - dofm) - 1) * ((3 - dofn) - 1);

  return chiSm;
  // return tally[0][0];
}

__device__ void noStates(const int idx, const int noGenes, int samples1,
                         int samples2, int *data1, int *data2, int *out1,
                         int *out2) {
  int *dataIn;
  int samplesIn;
  int start;
  int stop;
  int retVal;

  if (idx < noGenes) {
    dataIn = data1;
    samplesIn = samples1;
    start = idx * samplesIn;
    stop = start + samplesIn;
    assert(stop <= noGenes * samples1);
  } else {
    dataIn = data2;
    samplesIn = samples2;
    start = (idx - noGenes) * samplesIn;
    stop = start + samplesIn;
    assert(stop <= noGenes * samples2);
  }

  unsigned short statedata1[3] = {0, 0, 0};

  for (int i = start; i < stop; i++) {

    if (dataIn[i] == -1) {
      statedata1[0]++;
    }
    if (dataIn[i] == 0) {
      statedata1[1]++;
    }
    if (dataIn[i] == 1) {
      statedata1[2]++;
    }
  }

  retVal = 3;
  for (int i = 0; i < 3; i++) {
    out2[idx * 3 + i] = 1;
    if (statedata1[i] == 0) {
      out2[idx * 3 + i] = 0;

      retVal--;
    }
  }

  out1[idx] = retVal;
}

__global__ void run2(const int noGenes, const int leng, const int lengb,
                     int *tary, int *taryb, int *spacr, int *ff, int *dofout,
                     int *ppn, int *stf, int *out, int c, int *priorMatrix,
                     double alphaEdgePrior, double alphaEdge,
                     bool flag_pAdjust) {

  int index = threadIdx.x + blockDim.x * blockIdx.x; // global thread
  int tdx = threadIdx.x;                             // local thread
  int row = spacr[tdx];
  int col = ff[tdx];

  extern __shared__ int sharedMatrix[];

  *(sharedMatrix + row * noGenes + col) = *(priorMatrix + row * noGenes + col);
  __syncthreads();
  double edgeVal = 0; // stores chisquared value then stores gammds value- hold
                      // edge value to see if edge

  if (index < noGenes * 2) {
    // creates contingency tables
    noStates(index, noGenes, leng, lengb, tary, taryb, ppn, stf);
  }

  if (blockIdx.x <= leng) {
    edgeVal = sumrtime(0, leng, tary, spacr, ff, dofout, tdx);
  } else {
    edgeVal = sumrtime(leng, lengb, taryb, spacr, ff, dofout, tdx);
  }

  // edgeVal: p value of ChiSq (edge significance)
  edgeVal = 1.0 - deviceGammds(edgeVal / 2.0, ((double)dofout[index]) / 2.0);

  // Bonferroni correction
  if (flag_pAdjust) {
    edgeVal = min(1.0, edgeVal * (((noGenes - 1) * noGenes) / 2));
  }

  if (edgeVal < alphaEdge || (*(sharedMatrix + row * noGenes + col) == 1 &&
                              edgeVal < alphaEdgePrior)) {
    out[index] = 1;
  } else {
    out[index] = 0;
  }
}

__global__ void run2Scalable(const int noGenes, const int leng, const int lengb,
                             int *tary, int *taryb, int *spacr, int *ff,
                             int *dofout, int *ppn, int *stf, int *out, int c,
                             int *priorMatrix, double alphaEdgePrior,
                             double alphaEdge, bool flag_pAdjust, int BPN,
                             int TPB) {
  int netId = blockIdx.x / BPN;
  int localIdx = TPB * (blockIdx.x % BPN) + threadIdx.x;
  int globalIdx = localIdx + (netId * c);

  if (localIdx < c) {
    int row = spacr[localIdx];
    int col = ff[localIdx];
    double edgeVal = 0.0;

    if (globalIdx < noGenes * 2) {
      noStates(globalIdx, noGenes, leng, lengb, tary, taryb, ppn, stf);
    }

    // do we need a __syncthreads here?
    if (netId <= leng) {
      edgeVal = sumrtimeScalable(0, leng, tary, spacr, ff, dofout, localIdx,
                                 netId, globalIdx);
    } else {
      edgeVal = sumrtimeScalable(leng, lengb, taryb, spacr, ff, dofout,
                                 localIdx, netId, globalIdx);
    }

    edgeVal =
        1 - deviceGammds(edgeVal / 2.0, ((double)dofout[globalIdx]) / 2.0);

    // Bonferroni correction
    if (flag_pAdjust) {
      edgeVal = min(1.0, edgeVal * (((noGenes - 1) * noGenes) / 2));
    }

    if (edgeVal < alphaEdge || (*(priorMatrix + row * noGenes + col) == 1 &&
                                edgeVal < alphaEdgePrior)) {
      out[globalIdx] = 1;
    } else {
      out[globalIdx] = 0;
    }
  }
}
