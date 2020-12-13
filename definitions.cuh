/*
* definitions.cuh is the header file for kernel.cu and main.cu
* it also includes all the macros for both files
*/
#define MAX_THREADS 1024
#define MAX_GENES 50 //not used in any code
#ifdef _WIN32
const char DIR[20] = "PRIORS\\";
#endif
#ifdef linux
const char DIR[20] = "PRIORS/";
#include <errno.h>
#endif
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )



//CPU methods
static void HandleError(cudaError_t err, const char *file, int line);
void checkCUDAError(const char *msg);
void idPrep(int fixd, int combo, int *ary1, int *ary2);
double kool(double *P, double *Q, int scaler, int scaler1);
int structureUnique(int unique, int totEdges, int scaler1, int scaler2, int noGenes, bool *uniqueList, int *edgesPN, int *edgeAry, int *nodeAry, int *uniEdges, int *uniNodes, int *uniEpn);
double gammds(double x, double p, int *ifault);
int getMilliCount();
int getMilliSpan(int nTimeStart);
__host__ int arrayEqual(double *a, double *b, int size);
__host__ int cmpfunc(const void * a, const void * b);
__host__ void writeBdeuScores(char *outputFile, char *inputFile, char *classFile, char *genesetFile, char *class1, char *class2, int scaler, double *bdeuScores);
__host__ void writeNetworkFile(char *outputFile, char *inputFile, char *classFile, char *genesetFile, int networkNum, int *edgesPN,
	char genesetgenes[][40], int genesetlength, int *nodes, int *edges, int totalEdges, int *networkIds);
__host__ void writeEdgeListFile(char *outputFile, char *inputName, char *className, char *pathwayName, char genesetgenes[][40],
	int genesetlength, int *nodes, int *edges, int *edgesPN, int *priorMatrix, char class1[50], char class2[50]);
__host__ double mean(double *a, size_t size);
__host__ double variance(double mean, double *a, size_t size);
__host__ void checkParentLimit(int numNetworks, int numNodes, int maxParents, int *nodes, size_t size);



//CPU graph struct
typedef struct graphNode {
	int node;
	int edge;
	int inBoth;
} graphNode;




//GPU methods
__device__ void njLoop(int flag, int *stateSpaces, int node, int ssIdx, int noGenes,
	int samples, int noParents, int noParentsInstances, int *parentArray,
	int *stateFlags, int *data,
	int *stateVal, int *out, int *out2, int outOffSet, int out2Offset, int ppnLen);
__device__ double deviceGammds(double x, double p);
__device__ int aryCompare(int a, int b, int len, int *ary);
__device__ int edgeComparator(int *edgesPn, int *edgeAry, int tidx, int comp, int scaler1, int scaler2, int noEdges);
__device__  int pass1(int tidx, int noEdges, int scaler1, int scaler2, int gLength, int *dNodes, int *edgeAry, int *edgesPn, int shrunkAt, int *shnkplc);
__global__  void run25(int scaler1, int scaler2, int noEdges, int gLength, int scalerCombo, int *dedgesPN, int  *dNodes, int *dedgeAry, int *shrunk, int *scalerTest, int *shnkplc);
__device__  void edgePerNet(int *in, int *out, int *srchAry, int numNodes, const int maxParents, int c);
__device__ int nodeMaker(int c, int idx, int *data, int *srchAry);
__device__ void edgeCounter(int c, int *edgesPN, int index, int *data, int *nodeAry, int *srchAry, int *edgeAry, const int maxParents);
__device__ void nodeArrayParentCap(int numNodes, int *nodeArray, int *edgesPN, const int maxParents);
__global__ void run22(int c, int *edgesPN, int *data, int *nodeAry, int noGenes, int noEdges, int *srchAry, int *edgeAry, const int maxParents);
__device__ void space4(int scaler, int index, int noEdges, int *edgesPN, int *nodeAry, int *temp, int *edgeAry);
__device__ void reshape(int idblock, int noNodes, int *temp, int *out2);
__device__ void parentTrap(int scaler, int idblock, int index, int noEdges, int *edgesPN, int *out2, int *edgeAry, int *temp2);
__device__ void parentTrapPartduex(int scaler, int noEdges, int *edgesPN, int index, int noNodes, int *placeHolder, int *temp2, int *out2, int *out3);
__global__ void run3(int scaler, int *dedgesPN, int noEdges, int noNodes, int *temp, int *temp2, int *placeHolder, int *nodeAry, int *edgeAry, int *out2, int *out3);
__device__ double bDeu3(int flag, int blkoffset, int idxoffset, int tdxOffset, int *stateSpace, int *stateFlag, int *edgesPN, int idx, int noGenes, int noEdges, int uniSum, int samples, int nPrime, int *data, int *pEdges, int *pNodes, int *nij, int *nijk, int nijOs, int nijkOs, int ppnLen);
__global__ void run4(int scaler, int *dpedgesPN, int noGenes, int noEdges, int uniSum, int numClass1, int numClass2, int *data1, int *data2,
	int *pedges, int *pNodes, int *stateSpace, int *stateFlag, int *nij, int *nijk, double *out, int ppnLen);
__device__ double sumrtime(const int offset, const int len, int *data, int *spc, int *fr, int* dof, int idx);
__device__ double sumrtimeScalable(const int offset, const int len, int *data, int *spc, int *fr, int* dof, int idx, int netID, int globalIdx);
__device__ void noStates(const int idx, const int noGenes, int samples1, int samples2, int *data1, int *data2, int *out1, int *out2);
__global__ void run2(const int noGenes, const int leng, const int lengb, int *tary, int *taryb, int  *spacr, int *ff,
	int *dofout, int *ppn, int *stf, int *out, int c, int *priorMatrix, double threshpw, double thresh);
__global__ void run2Scalable(const int noGenes, const int leng, const int lengb, int *tary, int *taryb, int *spacr, int *ff,
				int *dofout, int *ppn, int *stf, int *out, int c, int *priorMatrix, double threshpw, double thresh, int BPN, int TPB);
__global__ void edgePerNetworkKernel(int *input, int *output, int *srchAry, int numNodes, const int maxParents, int c);
__global__ void noStates_kernel(const int noGenes, int samples1, int samples2, int *data1, int *data2, int *out1, int *out2);




















