#include "cuda_runtime.h"
#include "definitions.cuh"
#include "device_launch_parameters.h"
#include <assert.h>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cstdlib>
#include <cuda.h>
#include <math.h>
#include <math_functions.h>
#include <stdio.h>
#include <string.h>
#include <sys/timeb.h>
#include <time.h>
#include <vector>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void idPrep(int fixd, int combo, int *ary1, int *ary2) {

  int start = 2;
  int pos = 1;
  int pos2 = 0;
  for (int i = 0; i < combo; i++) {
    ary1[i] = pos;
    ary2[i] = pos2;
    pos++;
    if (pos == fixd) {
      pos = start;
      pos2++;
      start++;
    }
  }
}
// cooley function
double kool(double *P, double *Q, int scaler, int scaler1) {
  //	js = kool(lval1, sea, 0, scaler) / 2 + kool(lval1, sea, scaler, scaler)
  /// 2;
  int numberOfValue = scaler1;
  double D = 0;
  for (int i = 0; i < numberOfValue; i++) {
    if (Q[i] == 0) {
      continue;
    }

    if (P[i + scaler] == 0)
      continue;

    double temp = P[i + scaler] * log(P[i + scaler] / Q[i]);
    D += temp;
  }

  return D;
}

// creates unique required data for subsequent steps
int structureUnique(int unique, int totEdges, int scaler1, int scaler2,
                    int noGenes, bool *uniqueList, int *edgesPN, int *edgeAry,
                    int *nodeAry, int *uniEdges, int *uniNodes, int *uniEpn) {
  int start = 0;
  int start2 = 0;
  int set1 = 0;
  int count = 0;
  int count2 = 0;
  uniEpn[0] = 0;
  int place = 1;

  for (int i = 0; i < scaler2; i++) {
    if (uniqueList[i] == 1) {
      if (i < scaler1) {
        set1++;
      }
      // compute number of Edges in network
      int numEdges = 0;
      if (i == scaler2 - 1) {
        assert(i < scaler2 + 1);
        numEdges = totEdges - edgesPN[i];
      } else {
        assert(i < scaler2 + 1);
        numEdges = edgesPN[i + 1] - edgesPN[i];
      }
      // add matching values to new edge list
      int val;
      for (int j = 0; j < numEdges; j++) {
        assert(i < scaler2 + 1);
        val = edgeAry[edgesPN[i] + j];
        uniEdges[j + start] = val;
        count++;
      }

      start = count;
      // add matching values to new node list
      for (int k = 0; k < noGenes; k++) {
        uniNodes[k + start2] = nodeAry[i * noGenes + k];
        count2++;
      }
      start2 = count2;
      // add matching values to new master
      assert(place < unique + 1);
      /*if (place >= unique)
      {
              printf("place : %d\n", place);
      }*/
      uniEpn[place] = numEdges + uniEpn[place - 1];
      place++;
    }
  }
  return set1;
}
// gamma function
double gammds(double x, double p, int *ifault)

//****************************************************************************80
{
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
    *ifault = 1;
    value = 0.0;
    return value;
  }

  if (p <= 0.0) {
    *ifault = 1;
    value = 0.0;
    return value;
  }
  //
  //  LGAMMA is the natural logarithm of the gamma function.
  //
  arg = p * log(x) - lgamma(p + 1.0) - x;

  if (arg < log(uflo)) {
    value = 0.0;
    *ifault = 2;
    return value;
  }

  f = exp(arg);

  if (f == 0.0) {
    value = 0.0;
    *ifault = 2;
    return value;
  }

  *ifault = 0;
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

// the following two routines are used to benchmark time of execution
int getMilliCount() {
  timeb tb;
  ftime(&tb);
  int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
  return nCount;
}

int getMilliSpan(int nTimeStart) {
  int nSpan = getMilliCount() - nTimeStart;
  if (nSpan < 0)
    nSpan += 0x100000 * 1000;
  return nSpan;
}

__host__ int arrayEqual(double *a, double *b, int size) {
  int flag = 1;
  for (int i = 0; i < size; i++) {
    if (!(a[i] == b[i])) {
      flag = 0;
      return flag;
    }
  }

  return flag;
}

__host__ int cmpfunc(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

__host__ void writeBdeuScores(char *outputFile, char *inputFile,
                              char *classFile, char *genesetFile, char *class1,
                              char *class2, int scaler, double *bdeuScores) {
  FILE *scoreFile = fopen(outputFile, "w");
  if (scoreFile == NULL) {
    printf("scoreFile is NULL. Error code : %s. Exiting...\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  fprintf(scoreFile, "\t\t BDEU SCORES\n");
  fprintf(scoreFile, "input : %s\n", inputFile);
  fprintf(scoreFile, "class : %s\n", classFile);
  fprintf(scoreFile, "geneset : %s\n\n", genesetFile);
  fprintf(scoreFile, "Network \t  %s  \t   \t    %s\n", class1, class2);
  int networkIndex = 1;
  for (int j = 0; j < scaler; j++) {
    fprintf(scoreFile, "Network %d\t %f \t %f \n", networkIndex++,
            bdeuScores[j], bdeuScores[j + scaler]);
    // printf("lval1[%d] : %f\n lval1", j, lval1[j]);
  }
  fclose(scoreFile);
  printf("Finished bdeu file %s\n", outputFile);
}

__host__ void writeNetworkFile(char *outputFile, char *inputFile,
                               char *classFile, char *genesetFile,
                               int networkNum, int *edgesPN,
                               char genesetgenes[][MAX_LENGTH_NAME], int genesetlength,
                               int *nodes, int *edges, int totalEdges,
                               int *networkIds) {
  FILE *output = fopen(outputFile, "w");
  if (output == NULL) {
    printf("output when accessing networkOutput.txt is NULL. Error code : %s. "
           "Exiting...\n",
           strerror(errno));
    exit(EXIT_FAILURE);
  }

  fprintf(output, "%s", "\t\tUnique Networks\n\n");
  fprintf(output, "input : %s\n", inputFile);
  fprintf(output, "class : %s\n", classFile);
  fprintf(output, "geneset : %s\n\n\n", genesetFile);
  int edgePos = 0;

  // loop through all the unique networks
  for (int i = 0; i < networkNum; i++) {
    int total;
    if (i == networkNum - 1) {
      total = totalEdges - edgesPN[i];
    } else {
      total = edgesPN[i + 1] - edgesPN[i];
    }
    fprintf(output, "network %d (%d)\t number of edges : %d\n", i + 1,
            networkIds[i], total);
    fprintf(output, "%s", "---------------");
    fprintf(output, "%s", "---------------\n\n");
    if (total == 0) {
      continue;
    }
    // loop through each node in the ith network
    for (int j = 0; j < genesetlength - 1; j++) {
      // if change in node values --> edge has been found
      if (nodes[(i * genesetlength) + j + 1] !=
          nodes[(i * genesetlength) + j]) {
        int change =
            nodes[(i * genesetlength) + j + 1] - nodes[(i * genesetlength) + j];
        total -= change;
        for (int k = 0; k < change; k++) {
          fprintf(output, "%s - %s\n", genesetgenes[j],
                  genesetgenes[edges[edgePos++]]);
          // fprintf(output, "%d - %d ---> normal case total remaining : %d j :
          // %d\n", j, pUniEdges[edgePos++], total, j);
        }
      }
      if (j + 1 == genesetlength - 1 && total > 0) {
        // fprintf(output, "%s", "9s case\n");
        for (int k = 0; k < total; k++) {
          // fprintf(output, "%d - %d ---> 9s case\n", j + 1,
          // pUniEdges[edgePos++]);
          fprintf(output, "%s - %s\n", genesetgenes[j + 1],
                  genesetgenes[edges[edgePos++]]);
        }
      }
    }
    fprintf(output, "%s", "\n");
    fprintf(output, "%s", "\n");
  }

  fclose(output);
  printf("Finished writing %s\n", outputFile);
}

__host__ void writeEdgeListFile(char *outputFile, char *inputName,
                                char *className, char *pathwayName,
                                char genesetgenes[][MAX_LENGTH_NAME], int genesetlength,
                                int *nodes, int *edges, int *edgesPN,
                                int *priorMatrix, char class1[MAX_LENGTH_NAME],
                                char class2[MAX_LENGTH_NAME]) {
  FILE *output = fopen(outputFile, "w");
  // arrays to keep track of found nodes w/edges for each class
  std::vector<graphNode> class1Nodes;
  std::vector<graphNode> class2Nodes;

  // loop through *nodes/*edges to find get nodes and edge
  // store found nodes into either class1Nodes or class2Nodes with a graphNode
  // struct
  int edgePos = 0;
  char classes[2][MAX_LENGTH_NAME];
  strcpy(classes[0], class1);
  strcpy(classes[1], class2);
  // loop through all the unique networks
  for (int i = 0; i < 2; i++) {
    int total;
    if (i == 2 - 1) {
      total = edgesPN[2] - edgesPN[i];
    } else {
      total = edgesPN[i + 1] - edgesPN[i];
    }
    if (total == 0) {
      continue;
    }
    // loop through each node in the ith network
    for (int j = 0; j < genesetlength - 1; j++) {
      // if change in node values --> edge has been found
      if (nodes[(i * genesetlength) + j + 1] !=
          nodes[(i * genesetlength) + j]) {
        int change =
            nodes[(i * genesetlength) + j + 1] - nodes[(i * genesetlength) + j];
        total -= change;
        for (int k = 0; k < change; k++) {
          graphNode temp;
          temp.node = j;
          temp.edge = edges[edgePos++];
          temp.inBoth = 0;
          // fprintf(output, "%s - %s\n", genesetgenes[j],
          // genesetgenes[edges[edgePos++]]); printf("%s\t%s\t%s\n",
          // genesetgenes[j], genesetgenes[edges[edgePos++]], classes[i]);
          // printf("%s\n", classes[i]);
          if (strcmp(classes[i], class1) == 0) {
            class1Nodes.push_back(temp);
          } else {
            class2Nodes.push_back(temp);
          }
          // fprintf(output, "%d - %d ---> normal case total remaining : %d j :
          // %d\n", j, pUniEdges[edgePos++], total, j);
        }
      }
      if (j + 1 == genesetlength - 1 && total > 0) {
        for (int k = 0; k < total; k++) {
          graphNode temp;
          temp.node = j + 1;
          temp.edge = edges[edgePos++];
          temp.inBoth = 0;
          // fprintf(output, "%d - %d ---> 9s case\n", j + 1,
          // pUniEdges[edgePos++]); fprintf(output, "%s - %s\n", genesetgenes[j
          // + 1], genesetgenes[edges[edgePos++]]); printf("%s\t%s\t%s\n",
          // genesetgenes[j + 1], genesetgenes[edges[edgePos++]], classes[i]);
          // printf("%s\n", classes[i]);
          if (strcmp(classes[i], class1) == 0) {
            class1Nodes.push_back(temp);
          } else {
            class2Nodes.push_back(temp);
          }
        }
      }
    }
  }

  // determine which nodes in class1 are also in class2 - if the are mark inBoth
  // for each node so later we dont repeat when we print
  for (int i = 0; i < class1Nodes.size(); i++) {
    for (int j = 0; j < class2Nodes.size(); j++) {
      if (class1Nodes.at(i).node == class2Nodes.at(j).node &&
          class1Nodes.at(i).edge == class2Nodes.at(j).edge) {
        class1Nodes.at(i).inBoth = 1;
        class2Nodes.at(j).inBoth = 1;
      }
    }
  }

  // create spacer and ff rows to offset our reads for row/col access to the
  // prior matrix
  int c = ((genesetlength * genesetlength) - genesetlength) / 2;
  int position = 0;
  int *spacer = (int *)malloc(sizeof(int) * c);
  int *ff = (int *)malloc(sizeof(int) * c);
  for (int row = 1; row < genesetlength; row++) {
    for (int col = 0; col < row; col++) {
      spacer[position] = row;
      ff[position] = col;
      position++;
    }
  }

  // denote if eddy found an edge so that we can check later if the edge wasnt
  // found- 1 = EDDY found edge 0 = EDDY  did not find edge
  int *eddyFound = (int *)calloc(genesetlength * genesetlength, sizeof(int));

  // printf("Print contents of class1: \n");
  // for(int i = 0; i < class1Nodes.size(); i++)
  //{
  //	printf("%s\t%s\t in Both: %d\n", genesetgenes[class1Nodes.at(i).node],
  // genesetgenes[class1Nodes.at(i).edge], class1Nodes.at(i).inBoth);
  //}
  // printf("\nPrint contents of class2 : \n");
  // for(int i = 0; i < class2Nodes.size(); i++)
  //{
  //	printf("%s\t%s\t in Both: %d\n", genesetgenes[class2Nodes.at(i).node],
  // genesetgenes[class2Nodes.at(i).edge], class2Nodes.at(i).inBoth);
  //}

  char priorString[2][20] = {"NONE", "PRIOR"};
  // printf("ATTEMPTING FILE OUTPUT TEST: \n");
  for (int i = 0; i < class1Nodes.size(); i++) {

    int row = class1Nodes.at(i).node;
    int col = class1Nodes.at(i).edge;
    if (class1Nodes.at(i).inBoth == 1) {
      // if node is in both classes print to file with "Both" class specifier
      fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col],
              "Both", priorString[*(priorMatrix + row * genesetlength + col)]);
      // printf("%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col],
      // "Both", priorString[*(priorMatrix + row * genesetlength + col)]);
      // denote that eddy found this edge
      *(eddyFound + row * genesetlength + col) = 1;
      *(eddyFound + col * genesetlength + row) = 1;
      // printf("IN BOTH!");
    } else {
      // if it isnt both class write all class1 nodes
      fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col],
              class1, priorString[*(priorMatrix + row * genesetlength + col)]);
      // denote that eddy found this edge
      *(eddyFound + row * genesetlength + col) = 1;
      *(eddyFound + col * genesetlength + row) = 1;
    }
  }

  for (int i = 0; i < class2Nodes.size(); i++) {
    int row = class2Nodes.at(i).node;
    int col = class2Nodes.at(i).edge;
    if (class2Nodes.at(i).inBoth != 1) {
      // write to file all nodes as long as not also in class1 - that way we
      // dont repeat edges
      fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col],
              class2, priorString[*(priorMatrix + row * genesetlength + col)]);
      // printf("%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col],
      // class2, priorString[*(priorMatrix + row * genesetlength + col)]);
      // denote that eddy found an edge
      *(eddyFound + row * genesetlength + col) = 1;
      *(eddyFound + col * genesetlength + row) = 1;
    }
  }

  // write edges that werent found by EDDY
  // loop through prior matrix- if relationship isnt already found by eddy but
  // in the prior file, write relationship to file
  for (int i = 0; i < c; i++) {
    int row = spacer[i];
    int col = ff[i];
    // 0 in eddyFound = eddy did not find 1 in eddyFound = eddy did find
    if (*(eddyFound + row * genesetlength + col) == 0 &&
        *(priorMatrix + row * genesetlength + col) == 1) {
      fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col],
              "Neither", "PRIOR");
    }
  }

  free(spacer);
  spacer = NULL;
  free(ff);
  ff = NULL;
  free(eddyFound);
  eddyFound = NULL;

  fclose(output);
  printf("Finished writing edgelist file %s\n", outputFile);
}

__host__ double mean(double *a, size_t size) {
  double sum = 0.0;
  for (size_t i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum / size;
}

__host__ double variance(double mean, double *a, size_t size) {
  double sum = 0.0;
  for (size_t i = 0; i < size; i++) {
    sum += pow(a[i] - mean, 2);
  }
  return sum / size;
}

__host__ void checkParentLimit(int numNetworks, int numNodes, int maxParents,
                               int *nodes, size_t size) {
  // loop through each network
  for (int net = 0; net < numNetworks; net++) {
    // loop through nodes in each network
    for (int i = 0; i < numNodes - 1; i++) {
      if (nodes[(net * numNodes) + i + 1] - nodes[(net * numNodes) + i] >
          maxParents) {
        printf("PROBLEM @ %d %d\n", (net * numNodes) + i,
               (net * numNodes) + i + 1);
      }
    }
  }
}
