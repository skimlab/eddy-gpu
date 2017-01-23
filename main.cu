#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <math.h>
#include <string.h>
#include <cstdlib>
#include <sys/timeb.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#define MAX_THREADS 1024
//#define MAX_THREADS 23
#define MAX_GENES 50
#ifdef _WIN32
const char DIR[20] = "PRIORS\\";
#endif
#ifdef linux
const char DIR[20] = "PRIORS/";
#include <errno.h>
#endif
//extern "C"
//{
//#include "incomplete_beta_function.h"
//#include "beta_function.h"
//}
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
//void printVec(int** a, int n);

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg,
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
__device__ void njLoop(int flag, int *stateSpaces, int node, int ssIdx, int noGenes,
	int samples, int noParents, int noParentsInstances, int *parentArray,
	int *stateFlags, int *data,
	int *stateVal, int *out, int *out2, int outOffSet, int out2Offset, int ppnLen){
	int stateSs[3] = { -1, 0, 1 };
	int pid;
	//int kk;
	//int jj;
	//int countcount;
	int pidOffset;
	//int test = -99999;
	int *stateMatrixIdx;
	int *stateMatrixVal;



	stateMatrixIdx = (int *)malloc(sizeof(int)*noParents);
	stateMatrixVal = (int *)malloc(sizeof(int)*noParents);

	//routine for offset variable 
	if (flag == 0){
		pidOffset = 0;
	}
	else{
		pidOffset = noGenes;
	}
	//	test = pidOffset;




	for (int j = 0; j < noParentsInstances; j++){
		//test = j;
		out[j + outOffSet] = 0;


		int div = 1;

		//previously it was p2 < noParents - doesnt seem to be making a difference
		for (int p2 = 1; p2 < noParents; p2++){

			pid = parentArray[p2];
			//printf("pid + pidOffset = %d stateSpaceSize = %d", pid + pidOffset, stateSpaceSize);
			//printf("its working!!!!!!!!!!!\n");
			assert(pid + pidOffset < ppnLen);
			stateMatrixIdx[p2] = (j / div) % stateSpaces[pid + pidOffset]; //+pidOffset
			div *= stateSpaces[pid + pidOffset]; // +pidOffset
		}
		//prev p3 < noParents
		for (int p3 = 1; p3 < noParents; p3++){//

			pid = parentArray[p3];
			int stateCount = 0;
			for (int state = 0; state < 3; state++){
				
				if (stateFlags[pid * 3 + pidOffset * 3 + state] != 0){ //+pidOffset*3
					if (stateCount == stateMatrixIdx[p3]){
						stateMatrixVal[p3] = stateSs[state];
					}
					stateCount++;
				}
			}
		}


		assert(ssIdx < ppnLen);
		for (int k = 0; k < stateSpaces[ssIdx]; k++){



			out2[j*stateSpaces[ssIdx] + k + out2Offset] = 0;



			for (int s = 0; s < samples; s++){

				int	count = (data[node*samples + s] == stateVal[k]);

				for (int p3 = 1; p3 < noParents; p3++){
					pid = parentArray[p3];
					count *= (data[pid*samples + s] == stateMatrixVal[p3]);
				}

				out2[j*stateSpaces[ssIdx] + k + out2Offset] += count;
			}



			out[j + outOffSet] += out2[j*stateSpaces[ssIdx] + k + out2Offset];

		}//endkloop


	}



	free(stateMatrixIdx);
	free(stateMatrixVal);
	//get rid of return value
	//return test;

}

// preperation routine for indentifying unique datasets, used to create offsetted index mechanism
void idPrep(int fixd, int combo, int *ary1, int *ary2){
	int start = 2;
	int pos = 1;
	int pos2 = 0;
	for (int i = 0; i <combo; i++){
		ary1[i] = pos;
		ary2[i] = pos2;
		pos++;
		if (pos == fixd){
			pos = start;
			pos2++;
			start++;
		}
	}


}
//cooley function 
double kool(double *P, double *Q, int scaler, int scaler1) {
	//	js = kool(lval1, sea, 0, scaler) / 2 + kool(lval1, sea, scaler, scaler) / 2;
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
int structureUnique(int unique, int totEdges, int scaler1, int scaler2, int noGenes, bool *uniqueList, int *edgesPN, int *edgeAry, int *nodeAry, int *uniEdges, int *uniNodes, int *uniEpn){
	int start = 0;
	int start2 = 0;
	int set1 = 0;
	int count = 0;
	int count2 = 0;
	uniEpn[0] = 0;
	int place = 1;

	for (int i = 0; i < scaler2; i++){
		if (uniqueList[i] == 1){
			if (i < scaler1){
				set1++;
			}
			//compute number of Edges in network
			int numEdges = 0;
			if (i == scaler2 - 1){
				assert(i < scaler2 + 1);
				numEdges = totEdges - edgesPN[i];
			}
			else
			{
				assert(i < scaler2 + 1);
				numEdges = edgesPN[i + 1] - edgesPN[i];
			}
			//add matching values to new edge list
			int val;
			for (int j = 0; j < numEdges; j++){
				assert(i < scaler2 + 1);
				val = edgeAry[edgesPN[i] + j];
				uniEdges[j + start] = val;
				count++;
			}

			start = count;
			//add matching values to new node list
			for (int k = 0; k < noGenes; k++){
				uniNodes[k + start2] = nodeAry[i*noGenes + k];
				count2++;
			}
			start2 = count2;
			//add matching values to new master 
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

//device gamma function
//old version had *ifault as third parameter- look to CPU version
__device__ double deviceGammds(double x, double p)
{
	double a;
	double arg;
	double c;
	double e = 1.0E-09;
	double f;
	//int ifault2;
	double uflo = 1.0E-37;
	double value;
	//
	//  Check the input.
	//
	if (x <= 0.0)
	{
		//*ifault = 1;
		value = 0.0;
		return value;
	}

	if (p <= 0.0)
	{
		//*ifault = 1;
		value = 0.0;
		return value;
	}
	//
	//  LGAMMA is the natural logarithm of the gamma function.
	//
	arg = p * log(x) - lgamma(p + 1.0) - x;

	if (arg < log(uflo))
	{
		value = 0.0;
		//*ifault = 2;
		return value;
	}

	f = exp(arg);

	if (f == 0.0)
	{
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

	for (;;)
	{
		a = a + 1.0;
		c = c * x / a;
		value = value + c;

		if (c <= e * value)
		{
			break;
		}
	}

	value = value * f;

	return value;
	/*JR*/
}

//gamma function
double gammds(double x, double p, int *ifault)

//****************************************************************************80
{
	double a;
	double arg;
	double c;
	double e = 1.0E-09;
	double f;
	//int ifault2;
	double uflo = 1.0E-37;
	double value;
	//
	//  Check the input.
	//
	if (x <= 0.0)
	{
		*ifault = 1;
		value = 0.0;
		return value;
	}

	if (p <= 0.0)
	{
		*ifault = 1;
		value = 0.0;
		return value;
	}
	//
	//  LGAMMA is the natural logarithm of the gamma function.
	//
	arg = p * log(x) - lgamma(p + 1.0) - x;

	if (arg < log(uflo))
	{
		value = 0.0;
		*ifault = 2;
		return value;
	}

	f = exp(arg);

	if (f == 0.0)
	{
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

	for (;;)
	{
		a = a + 1.0;
		c = c * x / a;
		value = value + c;

		if (c <= e * value)
		{
			break;
		}
	}

	value = value * f;

	return value;
	/*JR*/
}





//simple routine to compare two arrays
__device__ int aryCompare(int a, int b, int len, int *ary){
	int flag = 0; //==

	for (int i = 0; i < len; i++){
		if (ary[a + i] != ary[b + i]){
			flag = 1; //!=
			break;
		}

	}
	return flag;

}

//compares two subsets of a given edge array and tests if they are equal
__device__ int edgeComparator(int *edgesPn, int *edgeAry, int tidx, int comp, int scaler1, int scaler2, int noEdges){
	int flag = 0; //denotes ==
	int eLen;
	if ((tidx == scaler2 - 1)){
		assert(tidx < scaler2 + 1);
		eLen = noEdges - edgesPn[tidx];
	}
	else
	{
		assert(tidx + 1 < scaler2 + 1);
		eLen = edgesPn[tidx + 1] - edgesPn[tidx];
	}

	int compLen;
	if (comp == scaler2 - 1){
		assert(comp < scaler2 + 1);
		compLen = noEdges - edgesPn[comp];
	}
	else
	{
		assert(comp + 1  <scaler2 + 1);
		compLen = edgesPn[comp + 1] - edgesPn[comp];
	}

	if (eLen != compLen){
		flag = 1;
		return flag;

	}
	//POSSIBLE BUG
	else{
		/*if (eLen > 3){
		eLen = 3;
		}*/

		flag = aryCompare(edgesPn[tidx], edgesPn[comp], eLen, edgeAry);
		return flag;
		//return eLen;
	}






}

// routines houses nodelist and edge list compare mechanisms used to identify unique information
__device__  int pass1(int tidx, int noEdges, int scaler1, int scaler2, int gLength, int *dNodes, int *edgeAry, int *edgesPn, int shrunkAt, int *shnkplc){

	//NodePass


	int flag = 0;
	flag = aryCompare(shnkplc[tidx] * gLength, shrunkAt*gLength, gLength, dNodes);
	if (flag == 0){
		flag = edgeComparator(edgesPn, edgeAry, shnkplc[tidx], shrunkAt, scaler1, scaler2, noEdges);
	}
	return flag;
	//return edgesPn[shnkplc[tidx]];
}


//GPU launch for pass1
__global__  void run25(int scaler1, int scaler2, int noEdges, int gLength, int scalerCombo, int *dedgesPN, int  *dNodes, int *dedgeAry, int *shrunk, int *scalerTest, int *shnkplc){
	int index = threadIdx.x + blockDim.x*blockIdx.x;

	//do we need this
	if (index < scalerCombo){

		shrunk[index] = pass1(index, noEdges, scaler1, scaler2, gLength, dNodes, dedgeAry, dedgesPN, scalerTest[index], shnkplc);

	}

}

//given an index, sums up number of binary data to count the number of edges in a network
//first parameter used to be int blockID
__device__  void edgePerNet(int *in, int *out, int *srchAry, int numNodes, const int maxParents, int c){

	int sum = 0;
	for (int k = 1; k <= blockIdx.x; k++)
	{
		for (int i = 1; i <numNodes; i++)
		{
			int start = srchAry[i - 1];
			int stop = srchAry[i];
			int nodeSum = 0;
			for (int j = start; j < stop; j++)
			{
				if (in[j + ((k - 1) * c)] == 1)
				{
					nodeSum++;
				}
			}
			if (nodeSum > maxParents)
			{
				nodeSum = maxParents;
			}
			sum += nodeSum;
		}
	}


	out[blockIdx.x] = sum;

	/*int start = 0;
	int stop = blockID;
	int sum = 0;
	for (int i = start; i < stop; i++){
	if (in[i] == 1){
	sum++;
	}
	}*/

	out[blockIdx.x] = sum;
	/*JR*/
}


//simple routine to get the sum a given row of data.
//
//__device__ void rowSum(int stop, int nogenes, int *out1){
//	if (stop == 0){
//		out1[stop] = 0;
//	}
//	else{
//		int dif = 0;
//		for (int i = 1; i < stop + 1; i++)
//		{
//
//
//
//			dif += nogenes - i;
//
//
//
//
//		}
//	}
//
//	/*JR*/
//}

// counts number of 1's in a dataset using an offsetted index
//__device__ void oneCounter(int c, int tdx, int *in, int *nodeAry, int *out){
//	if (tdx == 0){
//		out[tdx] = 0;
//	}
//	else
//	{
//		out[tdx] = -9999999999;
//		int sum = 0;
//		for (int i = 0; i <nodeAry[tdx]; i++)
//		{
//			if (in[i + blockIdx.x*c] == 1){
//				sum++;
//			}
//		}
//		out[tdx] = sum;
//	}
//	/*JR, GS*/
//}

//creates a node list using the following convention... data[i+1]-data[i] //const int maxParents
__device__ int nodeMaker(int c, int idx, int *data, int *srchAry){
	int sum = 0;
	//(threadIdx.x == 0 || threadIdx.x == 1)
	if (threadIdx.x == 0){
		return sum;
	}
	//shared memory code
	//else{ //srchAry[threadIdx.x-1]
	//	for (int i = 0; i < srchAry[threadIdx.x - 1]; i++){
	//		if (data[i] == 1){
	//			sum++;
	//		}
	//	}

	//	return sum;
	//}

	//normal non shared memory code
	else{ //srchAry[threadIdx.x-1]
		for (int i = 0; i < srchAry[threadIdx.x - 1]; i++){
			if (data[i + blockIdx.x * c] == 1){
				sum++;
			}
		}

		return sum;
	}
}

//creates an edge array using the following conventions																	//maxParents-const int
__device__ void edgeCounter(int c, int *edgesPN, int index, int *data, int *nodeAry, int *srchAry, int *edgeAry, const int maxParents){

	int place = (edgesPN[blockIdx.x] + nodeAry[index]);

	int pos = 0;
	int parentCounter = 0;

	int start = srchAry[threadIdx.x - 1];
	int stop = srchAry[threadIdx.x];
	//normal non shared memory code
	for (int i = start; i < stop; i++){
		if (parentCounter >= maxParents)
		{
			break;
		}
		if (data[blockIdx.x*c + i] == 1){
			edgeAry[place] = pos;
			place++;
			parentCounter++;

		}


		pos++;
	}
	//shared memory code
	/*
	for (int i = start; i < stop; i++){
	if (parentCounter >= maxParents)
	{
	break;
	}
	if (data[i] == 1){
	edgeAry[place] = pos;
	place++;
	parentCounter++;

	}


	pos++;
	}*/
	/*JR, GS*/
}



//limits the nodes to have maxParents edges
__device__ void nodeArrayParentCap(int numNodes, int *nodeArray, int *edgesPN, const int maxParents)
{

	//get starting place in node array to correct
	int start = blockIdx.x * blockDim.x;
	int stop = start + numNodes - 1;
	//only have one thread operate per block
	if (threadIdx.x == 0)
	{
		for (int i = start; i < stop; i++)
		{

			if ((nodeArray[i + 1] - nodeArray[i]) > maxParents)
			{
				int temp = nodeArray[i + 1];
				nodeArray[i + 1] = nodeArray[i] + maxParents;
				int decrement = temp - nodeArray[i + 1];
				//should this be <= or <???
				for (int j = i + 2; j <= stop; j++)
				{
					nodeArray[j] = nodeArray[j] - decrement;
				}
			}
		}
	}
}



//launch for parent node graph data structure creation
__global__ void run22(int c, int *edgesPN, int *data, int *nodeAry, int noGenes, int noEdges, int *srchAry, int *edgeAry, const int maxParents){

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	/*__shared__ int sharedData[45];

	if (threadIdx.x == 0)
	{
	memcpy(sharedData, &data[blockIdx.x * c], c * sizeof(int));
	}
	__syncthreads();*/

	nodeAry[index] = nodeMaker(c, index, data, srchAry);//-normal code
	//nodeAry[index] = nodeMaker(c, index, sharedData, srchAry); //-shared memory code

	__syncthreads();
	// oneCounter(c, threadIdx.x, data, nodeAry, placeHolder);

	nodeArrayParentCap(noGenes, nodeAry, edgesPN, maxParents);

	__syncthreads();

	if (threadIdx.x > 0){ edgeCounter(c, edgesPN, index, data, nodeAry, srchAry, edgeAry, maxParents); }//normal code
	//if (threadIdx.x > 0){ edgeCounter(c, edgesPN, index, data, nodeAry, srchAry, edgeAry); }
	//if (threadIdx.x > 0){ edgeCounter(c, edgesPN, index, sharedData, nodeAry, srchAry, edgeAry, maxParents); } //-shared memory code



	/*JR, GS*/
}


//creates required space index for subsequent procedures in parentTrap routine
__device__ void space4(int scaler, int index, int noEdges, int *edgesPN, int *nodeAry, int *temp, int *edgeAry){
	int sum = 0;
	int find = threadIdx.x;
	int start = edgesPN[blockIdx.x];
	int stop = edgesPN[blockIdx.x + 1];
	if (blockIdx.x == scaler - 1){ //22251244
		stop = noEdges;
	}


	for (int i = start; i < stop; i++){
		if (edgeAry[i] == find){
			sum++;
		}
	}
	temp[index] = sum;


	/*JR*/
}

//restructures passed in temporary dataset for parent Trap routine
__device__ void reshape(int idblock, int noNodes, int *temp, int *out2){
	int count = 0;
	int start = idblock*noNodes;
	for (int j = start + 1; j < start + noNodes; j++){
		count = temp[j - 1] + count;
		out2[j] = count;
	}
	out2[start] = 0;

}
//creates second place holder array before final parent array
__device__ void parentTrap(int scaler, int idblock, int index, int noEdges, int *edgesPN, int *out2, int *edgeAry, int *temp2){
	int finder = threadIdx.x;
	int incep = out2[index] + edgesPN[blockIdx.x];
	if (blockIdx.x == 0){
		//	incep = out2[finder];
	}
	int start = edgesPN[blockIdx.x];
	int stop = edgesPN[blockIdx.x + 1];
	if (blockIdx.x == scaler - 1){
		stop = noEdges;
	}


	for (int i = start; i < stop; i++){


		if (edgeAry[i] == finder){

			temp2[incep] = i;
			incep++;
		}


	}
}

//creates final parent data using previous place holder values
__device__ void parentTrapPartduex(int scaler, int noEdges, int *edgesPN, int index, int noNodes, int *placeHolder, int *temp2, int *out2, int *out3){
	int start = out2[index] + edgesPN[blockIdx.x];
	int start2 = blockIdx.x*blockDim.x;
	//int posr = 1;

	int stop = out2[index + 1] + edgesPN[blockIdx.x];

	if (threadIdx.x == noNodes - 1){
		stop = edgesPN[blockIdx.x + 1];
	}
	if (blockIdx.x == scaler - 1){
		stop = noEdges;
	}
	for (int i = start; i < stop; i++)
	{
		for (int j = start2; j < start2 + noNodes; j++)
		{
			if (temp2[i] >= placeHolder[j] + edgesPN[blockIdx.x] && temp2[i] <placeHolder[j + 1] + edgesPN[blockIdx.x])
			{

				out3[i] = j - blockDim.x*blockIdx.x;

			}

		}


	}


}



// luanch for different parent information extracting routines
__global__ void run3(int scaler, int *dedgesPN, int noEdges, int noNodes, int *temp, int *temp2, int *placeHolder, int *nodeAry, int *edgeAry, int *out2, int *out3){
	int index = threadIdx.x + blockDim.x*blockIdx.x;


	space4(scaler, index, noEdges, dedgesPN, nodeAry, temp, edgeAry);

	if (index < noEdges && index>-1){
		temp2[index] = 0;
	}
	__syncthreads();
	reshape(blockIdx.x, noNodes, temp, out2);



	//out3[index] = 0;
	//out2[index] = 0;
	__syncthreads();

	parentTrap(scaler, blockIdx.x, index, noEdges, dedgesPN, out2, edgeAry, temp2);

	__syncthreads();

	//	__syncthreads();

	parentTrapPartduex(scaler, noEdges, dedgesPN, index, noNodes, placeHolder, temp2, out2, out3);

	__syncthreads();

}

// routine counts different states(-1,0,1) of a given dataset.
/*

__device__ void valcounter(int index, int numClass, int *data, int *out){
int start = index*numClass;
int pos = index * 3;
int stop = start + numClass;
for (int i = start; i < stop; i++)
{
if (data[i] == -1){
out[pos]++;
}
if (data[i] == 0){
out[pos + 1]++;
}
if (data[i] == 1){
out[pos + 2]++;
}
}

}

*/

//bdeu function version 3, see http://link.springer.com/article/10.1186/1471-2105-13-S15-S14/fulltext.html for more info
__device__ double bDeu3(int flag, int blkoffset, int idxoffset, int tdxOffset, int *stateSpace, int *stateFlag, int *edgesPN, int idx, int noGenes, int noEdges, int uniSum, int samples, int nPrime, int *data, int *pEdges, int *pNodes, int *nij, int *nijk, int nijOs, int nijkOs, int ppnLen){
	//int tidx = threadIdx.x;
	int noParents = 0;
	//int *statedata1;
	//int *statedata2;
	//int test = 0;
	int pidOffset = noGenes;
	if (flag == 0){
		pidOffset = 0;
	}



	int noEdges1 = 0;


	if (blockIdx.x == uniSum - 1 || blockIdx.x == uniSum * 2 - 1){
		noEdges1 = noEdges - edgesPN[blockIdx.x - blkoffset];
	}
	else
	{
		noEdges1 = edgesPN[blockIdx.x + 1 - blkoffset] - edgesPN[blockIdx.x - blkoffset];
	}









	//divergence?
	if (threadIdx.x == noGenes - 1){
	
		noParents = 1 + noEdges1 - pNodes[idx - idxoffset];
		if(noParents > 4 )
			printf("Edge Case\n");
	}
	else
	{
		noParents = 1 + (pNodes[idx + 1 - idxoffset] - pNodes[idx - idxoffset]);
		if(noParents > 4)
		{
			printf("normal case\n");
		}
	}
	if (noParents > 4){
		noParents = 4;
		printf("parents triggered idx : %d %d\n", idx, noEdges1);

	}

	//test = noParents;


	int pos2 = 1; //test
	int *parentAry;

	//parent scale variable, an attempt to make noParent >1 scalable to all cases

	parentAry = (int*)malloc(sizeof(int) * (noParents));
	parentAry[0] = threadIdx.x; //test


	//where parentarray locations are in data
	for (int k = 0; k < noParents - 1; k++)
	{
		int kk = pNodes[idx - idxoffset] + edgesPN[blockIdx.x - blkoffset] + k;
		parentAry[pos2] = 0;
		parentAry[pos2] = pEdges[kk];
		pos2++;
	}


	//	int *nij; int *nijk;
	int *stateVal;
	int noParentInstances = 1;
	int pid;

	//state space and state flag come from first kernel run2
	assert(tdxOffset + threadIdx.x < ppnLen);
	stateVal = (int *)malloc(sizeof(int) * stateSpace[tdxOffset + threadIdx.x]); //ask
	int pos = 0;
	int stateSs[3] = { -1, 0, 1 };
	for (int i = 0; i < 3; i++){
		if (stateFlag[(3 * tdxOffset + (3 * threadIdx.x)) + i] == 1){
			stateVal[pos] = stateSs[i];
			pos++;
		}
	}


	for (int i = 1; i < noParents; i++){
		pid = parentAry[i];
		assert(pid + pidOffset < ppnLen);
		noParentInstances *= stateSpace[pid + pidOffset];
	}
	//test = noParentInstances;

	//	nij = (int *)malloc(sizeof(int)*noParentInstances);

	//	nijk = (int *)malloc(sizeof(int)*noParentInstances*stateSpace[tdxOffset + threadIdx.x]);


	//test = threadIdx.x+blockDim.x*blockIdx.x;

	//	 flag, int *stateSpaces, int node, int ssIdx, int blkId, int noGenes,int samples, int noParents, int noParentsInstances, int *parentArray,int *stateFlags, int *data,int *stateVal, int *stateMatrixIdx, int *stateMatrixVal, int *out, int *out2)



	//test = njLoop(flag, stateSpace, threadIdx.x, tdxOffset + threadIdx.x, noGenes, samples, noParents, noParentInstances, parentAry, stateFlag, data, stateVal, nij, nijk, nijOs, nijkOs);
	njLoop(flag, stateSpace, threadIdx.x, tdxOffset + threadIdx.x, noGenes, samples, noParents, noParentInstances, parentAry, stateFlag, data, stateVal, nij, nijk, nijOs, nijkOs, ppnLen);


	free(parentAry);
	free(stateVal);



	double sum1 = 0;
	double sum2 = 0;

	for (int j = 0; j < noParentInstances; j++){

		sum1 = 0;
		assert(tdxOffset + threadIdx.x < ppnLen);
		for (int k = 0; k <stateSpace[tdxOffset + threadIdx.x]; k++){
			sum1 += lgamma(double(nijk[j*stateSpace[threadIdx.x + tdxOffset] + k + nijkOs]) + double(nPrime) / double(noParentInstances*stateSpace[threadIdx.x + tdxOffset]))
				- lgamma(double(nPrime) / double(noParentInstances*stateSpace[threadIdx.x + tdxOffset]));
		}
		sum2 += lgamma(double(nPrime) / double(noParentInstances)) - lgamma(double(nij[j + nijOs]) + double(nPrime) / double(noParentInstances)) + sum1;
	}







	//sum2 = nijk[1];
	//	sum2 = 0;

	//	int sum2 = 0;
	//	for (int i = 0; i < stateSpace[threadIdx.x + tdxOffset] * noParentInstances; i++){
	//			sum2 += nijk[i];
	//	}


	//sum2 = nijk[4];

	//	sum2 = noParentInstances;


	//sum1 = double(xax[2]);

	//return test same result as return sum2
	return sum2;
}

//launch for bdeu routine, split by offsetted data (sum of unique networks multiplied by the number of genes
__global__ void run4(int scaler, int *dpedgesPN, int noGenes, int noEdges, int uniSum, int numClass1, int numClass2, int *data1, int *data2,
	int *pedges, int *pNodes, int *stateSpace, int *stateFlag, int *nij, int *nijk, double *out, int ppnLen){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	//printf("@@ \n %d %d ", index, data[index]);
	


	if (index < uniSum*noGenes){
		out[index] = bDeu3(0, 0, 0, 0, stateSpace, stateFlag, dpedgesPN, index, noGenes, noEdges, uniSum, numClass1, 4, data1, pedges, pNodes, nij, nijk, index * 27, index * 81, ppnLen);
	}

	else{
		out[index] = bDeu3(1, uniSum, (uniSum)*noGenes, noGenes, stateSpace, stateFlag, dpedgesPN, index, noGenes, noEdges, uniSum, numClass2, 4, data2, pedges, pNodes, nij, nijk, index * 27, index * 81, ppnLen);
	}
}



// a tally routine for each of the state premutations by the two datasets, next using that information 
//to calc expected dataset used finally for a chi-square routine which is the return value.
// prior parameters included int offset and int lenLim- not used so deleted
__device__ double sumrtime(const int offset, const int len, int *data, int *spc, int *fr, int* dof, int idx){

	int skipper, con = 3;
	//contigency table observed
	int tally[3][3] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	//contigency table expected
	double expected[3][3] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	skipper = blockIdx.x - offset;
	for (int k1 = 0; k1 < len; k1++){
		if ((skipper != 0) && (k1 == skipper - 1)){
			continue;
		}
		//place tally for each occurence in observed contingency table
		if ((data[(spc[idx] * len) + k1] == -1) && (data[(fr[idx] * len) + k1] == -1)){
			tally[0][0]++;
		}
		else if ((data[spc[idx] * len + k1] == -1) && (data[fr[idx] * len + k1] == 0)){
			tally[0][1]++;
		}
		else if ((data[spc[idx] * len + k1] == -1) && (data[fr[idx] * len + k1] == 1)){
			tally[0][2]++;
		}
		else if ((data[spc[idx] * len + k1] == 0) && (data[fr[idx] * len + k1] == -1)){
			tally[1][0]++;
		}
		else if ((data[spc[idx] * len + k1] == 0) && (data[fr[idx] * len + k1] == 0)){
			tally[1][1]++;
		}
		else if ((data[spc[idx] * len + k1] == 0) && (data[fr[idx] * len + k1] == 1)){
			tally[1][2]++;
		}
		else if ((data[spc[idx] * len + k1] == 1) && (data[fr[idx] * len + k1] == -1)){
			tally[2][0]++;
		}
		else if ((data[spc[idx] * len + k1] == 1) && (data[fr[idx] * len + k1] == 0)){
			tally[2][1]++;
		}
		else if ((data[spc[idx] * len + k1] == 1) && (data[fr[idx] * len + k1] == 1)){
			tally[2][2]++;
		}
	}

	//summation of rows and columns for chi squared table
	int ex[7] = { 0, 0, 0, 0, 0, 0, 0 };
	double yates = 0;
	for (int c = 0; c < con; c++){
		for (int c1 = 0; c1 < con; c1++){
			if (c1 == 0){
				ex[0] += tally[c][c1];
			}
			else if (c1 == 1){
				ex[1] += tally[c][c1];
			}
			else if (c1 == 2){
				ex[2] += tally[c][c1];
			}
		}
		for (int b = 0; b < con; b++){
			if (b == 0){
				ex[3] += tally[b][c];
			}
			else if (b == 1){
				ex[4] += tally[b][c];
			}
			else if (b == 2){
				ex[5] += tally[b][c];
			}
		}
	}

	if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5]))
	{
		printf("bad math!!!!!!!!");
	}
	else
	{
		ex[6] = ex[0] + ex[1] + ex[2];
		//printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d %d %d %d***", idx, tally[0][0], tally[0][1], tally[0][2], tally[1][0], 
		//tally[1][1], tally[1][2], tally[2][0], tally[2][1], tally[2][2], ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
	}
	double divisor = double(ex[6]);
	for (int c = 0; c < con; c++){
		for (int c1 = 0; c1 < con; c1++){
			expected[c][c1] = (double(ex[c1])*double(ex[c + 3]) / divisor);
		}
	}

	//set use of yates correction if 1 cell < 5
	int flag = 0;
	for (int c = 0; c < con; c++){
		for (int c1 = 0; c1 < con; c1++){
			if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))){
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

	//calculating chi squared sum

	for (int ii = 0; ii < 3; ii++){
		if (ex[ii] == 0){
			dofm++;
		}
		if (ex[ii + 3] == 0){
			dofn++;
		}
		for (int jj = 0; jj < 3; jj++)
		{
			//save calculation time if not zero
			if ((ex[jj] * ex[ii + 3]) != 0){


				chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) / expected[ii][jj];
			}

		}
	}



	//
	dof[threadIdx.x + blockDim.x*blockIdx.x] = ((3 - dofm) - 1)*((3 - dofn) - 1);







	return chiSm;
	//return tally[0][0];
}

__device__ double sumrtimeScalable(const int offset, const int len, int *data, int *spc, int *fr, int* dof, int idx, int netID, int globalIdx){

	int skipper, con = 3;
	//contigency table observed
	int tally[3][3] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	//contigency table expected
	double expected[3][3] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	//skipper = blockIdx.x - offset;
	skipper = netID - offset;
	for (int k1 = 0; k1 < len; k1++){
		if ((skipper != 0) && (k1 == skipper - 1)){
			continue;
		}
		//place tally for each occurence in observed contingency table
		if ((data[(spc[idx] * len) + k1] == -1) && (data[(fr[idx] * len) + k1] == -1)){
			tally[0][0]++;
		}
		else if ((data[spc[idx] * len + k1] == -1) && (data[fr[idx] * len + k1] == 0)){
			tally[0][1]++;
		}
		else if ((data[spc[idx] * len + k1] == -1) && (data[fr[idx] * len + k1] == 1)){
			tally[0][2]++;
		}
		else if ((data[spc[idx] * len + k1] == 0) && (data[fr[idx] * len + k1] == -1)){
			tally[1][0]++;
		}
		else if ((data[spc[idx] * len + k1] == 0) && (data[fr[idx] * len + k1] == 0)){
			tally[1][1]++;
		}
		else if ((data[spc[idx] * len + k1] == 0) && (data[fr[idx] * len + k1] == 1)){
			tally[1][2]++;
		}
		else if ((data[spc[idx] * len + k1] == 1) && (data[fr[idx] * len + k1] == -1)){
			tally[2][0]++;
		}
		else if ((data[spc[idx] * len + k1] == 1) && (data[fr[idx] * len + k1] == 0)){
			tally[2][1]++;
		}
		else if ((data[spc[idx] * len + k1] == 1) && (data[fr[idx] * len + k1] == 1)){
			tally[2][2]++;
		}
	}

	//summation of rows and columns for chi squared table
	int ex[7] = { 0, 0, 0, 0, 0, 0, 0 };
	double yates = 0;
	for (int c = 0; c < con; c++){
		for (int c1 = 0; c1 < con; c1++){
			if (c1 == 0){
				ex[0] += tally[c][c1];
			}
			else if (c1 == 1){
				ex[1] += tally[c][c1];
			}
			else if (c1 == 2){
				ex[2] += tally[c][c1];
			}
		}
		for (int b = 0; b < con; b++){
			if (b == 0){
				ex[3] += tally[b][c];
			}
			else if (b == 1){
				ex[4] += tally[b][c];
			}
			else if (b == 2){
				ex[5] += tally[b][c];
			}
		}
	}

	if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5]))
	{
		printf("bad math!!!!!!!!");
	}
	else
	{
		ex[6] = ex[0] + ex[1] + ex[2];
		//printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d %d %d %d***", idx, tally[0][0], tally[0][1], tally[0][2], tally[1][0], 
		//tally[1][1], tally[1][2], tally[2][0], tally[2][1], tally[2][2], ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
	}
	double divisor = double(ex[6]);
	for (int c = 0; c < con; c++){
		for (int c1 = 0; c1 < con; c1++){
			expected[c][c1] = (double(ex[c1])*double(ex[c + 3]) / divisor);
		}
	}

	//set use of yates correction if 1 cell < 5
	int flag = 0;
	for (int c = 0; c < con; c++){
		for (int c1 = 0; c1 < con; c1++){
			if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))){
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

	//calculating chi squared sum

	for (int ii = 0; ii < 3; ii++){
		if (ex[ii] == 0){
			dofm++;
		}
		if (ex[ii + 3] == 0){
			dofn++;
		}
		for (int jj = 0; jj < 3; jj++)
		{
			//save calculation time if not zero
			if ((ex[jj] * ex[ii + 3]) != 0){


				chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) / expected[ii][jj];
			}

		}
	}



	//
	//dof[threadIdx.x + blockDim.x*netID] = ((3 - dofm) - 1)*((3 - dofn) - 1);
	dof[globalIdx] = ((3 - dofm) - 1)*((3-dofn) - 1);	






	return chiSm;
	//return tally[0][0];
}

//routine calculates number of zeros in given state data, the two outputs are different representations of this information
//out1 is number of non-zeros in a given state dataset while out2 is the binary representation of this for the different states.
__device__ void noStates(const int idx, const int noGenes, int samples1, int samples2, int *data1, int *data2, int *out1, int *out2){
	int *dataIn;
	int samplesIn;
	int start;
	int stop;
	int retVal;


	if (idx < noGenes){
		dataIn = data1;
		samplesIn = samples1;
		start = idx*samplesIn;
		stop = start + samplesIn;
		assert(stop <= noGenes * samples1);
	}
	else{
		dataIn = data2;
		samplesIn = samples2;
		start = (idx - noGenes)*samplesIn;
		stop = start + samplesIn;
		assert(stop <= noGenes * samples2);
	}



	unsigned short statedata1[3] = { 0, 0, 0 };

	for (int i = start; i < stop; i++)
	{
	
		if (dataIn[i] == -1){
			statedata1[0]++;
		}
		if (dataIn[i] == 0){
			statedata1[1]++;

		}
		if (dataIn[i] == 1){
			statedata1[2]++;

		}

	}


	retVal = 3;
	for (int i = 0; i < 3; i++){
		out2[idx * 3 + i] = 1;
		if (statedata1[i] == 0){
			out2[idx * 3 + i] = 0;

			retVal--;
		}




	}
	
	out1[idx] = retVal;
}


//launch for nostate and summertime routine
//creates contigency tables + tallies chi squared results and run gammds calculation that gives if there is an edge
//with this edge data constructs a binary edge of edges in out
//previous parameter- tout --> dout2 on CPU side- only used to store chi squared calcs- instead just place in ones array
//and then take ones array value, use it to calculate gammds function and then place back in ones array
__global__ void run2(const int noGenes, const int leng, const int lengb, int *tary, int *taryb, int  *spacr, int *ff,
	int *dofout, int *ppn, int *stf, int *out, int c, int *priorMatrix, double pw){

	int index = threadIdx.x + blockDim.x*blockIdx.x; //global thread
	int tdx = threadIdx.x; //local thread
	int row = spacr[tdx];
	int col = ff[tdx];
	
	extern __shared__ int sharedMatrix[];
	
	*(sharedMatrix + row * noGenes + col) = *(priorMatrix + row * noGenes + col);
	__syncthreads();
	double edgeVal = 0; //stores chisquared value then stores gammds value- hold edge value to see if edge
	

	if (index < noGenes * 2){
		//creates contingency tables
		noStates(index, noGenes, leng, lengb, tary, taryb, ppn, stf);
	}


	//__syncthreads(); we dont need this right? - noStates doesnt change any values that are used in run2- 
	//affects ppn/stf- not used until run4

	if (blockIdx.x <= leng){
		//ones[threadIdx.x + blockDim.x*blockIdx.x] = sumrtime(0, leng, tary, spacr, ff, dofout, tdx);
		edgeVal = sumrtime(0, leng, tary, spacr, ff, dofout, tdx);
	}
	else{
		//ones[(threadIdx.x + blockDim.x*(blockIdx.x))] = sumrtime(leng, lengb, taryb, spacr, ff, dofout, tdx);
		
		edgeVal = sumrtime(leng, lengb, taryb, spacr, ff, dofout, tdx);
	}


	//__syncthreads();

	//gammds calculation to determine binary data
	//ones[index] = deviceGammds(((double)ones[index]) / 2, ((double)dofout[index]) / 2);
	edgeVal = deviceGammds(edgeVal / 2, ((double)dofout[index]) / 2);
	//if (ones[index] > .8)
	if (edgeVal > .8 || (*(sharedMatrix + row * noGenes + col) == 1 && edgeVal > .8 * pw))
	//if (edgeVal > .8)
	{
		out[index] = 1;
	}
	else
	{
		out[index] = 0;
	}
}

__global__ void run2Scalable(const int noGenes, const int leng, const int lengb, int *tary, int *taryb, int *spacr, int *ff,
							int *dofout, int *ppn, int *stf, int *out, int c, int *priorMatrix, double pw, int BPN, int TPB)
{
	int netId = blockIdx.x / BPN;
	int localIdx = TPB * (blockIdx.x % BPN) + threadIdx.x;
	int globalIdx = localIdx + (netId * c);
	
	if(localIdx < c)
	{
		int row = spacr[localIdx];
		int col = ff[localIdx];
		double edgeVal = 0.0;
	
		if(globalIdx < noGenes * 2)
		{
			noStates(globalIdx, noGenes, leng, lengb, tary, taryb, ppn, stf);
		}
		//do we need a __syncthreads here?
		if(netId <= leng)
		{
			edgeVal = sumrtimeScalable(0, leng, tary, spacr, ff, dofout, localIdx, netId, globalIdx);
		}
		else
		{
			edgeVal = sumrtimeScalable(leng, lengb, taryb, spacr, ff, dofout, localIdx, netId, globalIdx);
		}
	
		edgeVal = deviceGammds(edgeVal / 2, ((double)dofout[globalIdx]) / 2);
	
		if(edgeVal > .8 || (*(priorMatrix + row * noGenes + col) == 1 && edgeVal > .8 * pw))
		{
			out[globalIdx] = 1;
		}
		else
		{
			out[globalIdx] = 0;
		}
	}


}


//the following two routines are used to benchmark time of execution 
int getMilliCount(){
	timeb tb;
	ftime(&tb);
	int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
	return nCount;
}

int getMilliSpan(int nTimeStart){
	int nSpan = getMilliCount() - nTimeStart;
	if (nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}

__host__ int arrayEqual(double *a, double *b, int size)
{
	int flag = 1;
	for (int i = 0; i < size; i++)
	{
		if (!(a[i] == b[i]))
		{
			flag = 0;
			return flag;
		}
	}

	return flag;
}



__global__ void edgePerNetworkKernel(int *input, int *output, int *srchAry, int numNodes, const int maxParents, int c)
{

	//edgePerNet(input, output, srchAry, numNodes, maxParents, c);

	//compute sum of edges for a single network and store in output[] - prefix sum is completed on CPU	
	
	int sum = 0;
	for(int i = 1; i < numNodes; i++)
	{
		int start = srchAry[i-1];
		int stop = srchAry[i];
		int nodeSum = 0;
		
		for(int j = start; j < stop; j++)
		{
			if(input[j + (blockIdx.x * c)] == 1)
			{
				nodeSum++;
			}
		}
	
		if(nodeSum > maxParents)
		{
			nodeSum = maxParents;
		}
		
		sum += nodeSum;
	}

	output[blockIdx.x] = sum;







	/*
	extern __shared__ int dataMem[];
	__shared__ int searcher[sizeof(int) * MAX_GENES];
	memcpy(searcher, srchAry, sizeof(int) * numNodes);


	int sum = 0;
	for (int k = 1; k <= blockIdx.x; k++)
	{
		//sync threads to prevent race condition
		//some threads finish early and overwrite dataMem with next data chunk
		__syncthreads();
		dataMem[threadIdx.x] = input[((k - 1) * c) + threadIdx.x];
		__syncthreads();
		for (int i = 1; i <numNodes; i++)
		{
			//int start = srchAry[i - 1];
			//int stop = srchAry[i];
			int start = searcher[i - 1];
			int stop = searcher[i];
			int nodeSum = 0;
			for (int j = start; j < stop; j++)
			{
				//if (input[j + ((k - 1) * c)] == 1)
				if (dataMem[j] == 1)
				{
					nodeSum++;
				}
			}
			if (nodeSum > maxParents)
			{
				nodeSum = maxParents;
			}
			sum += nodeSum;
		}
	}
	output[blockIdx.x] = sum;
	*/

}





__host__ int cmpfunc(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}

__host__ void writeBdeuScores(char *outputFile, char *inputFile, char *classFile, char *genesetFile, char *class1, char *class2, int scaler, double *bdeuScores)
{
	FILE *scoreFile = fopen(outputFile, "w");
	if (scoreFile == NULL)	{ printf("scoreFile is NULL. Error code : %s. Exiting...\n", strerror(errno)); exit(EXIT_FAILURE); }
	fprintf(scoreFile, "\t\t BDEU SCORES\n");
	fprintf(scoreFile, "input : %s\n", inputFile);
	fprintf(scoreFile, "class : %s\n", classFile);
	fprintf(scoreFile, "geneset : %s\n\n", genesetFile);
	fprintf(scoreFile, "Network \t  %s  \t   \t    %s\n", class1, class2);
	int networkIndex = 1;
	for (int j = 0; j < scaler; j++)
	{
		fprintf(scoreFile, "Network %d\t %f \t %f \n", networkIndex++, bdeuScores[j], bdeuScores[j + scaler]);
		//printf("lval1[%d] : %f\n lval1", j, lval1[j]);
	}
	fclose(scoreFile);
	printf("Finished bdeu file %s\n", outputFile);
}

__host__ void writeNetworkFile(char *outputFile, char *inputFile, char *classFile, char *genesetFile, int networkNum, int *edgesPN,
	char genesetgenes[][40], int genesetlength, int *nodes, int *edges, int totalEdges, int *networkIds)
{
	FILE *output = fopen(outputFile, "w");
	if (output == NULL)	{ printf("output when accessing networkOutput.txt is NULL. Error code : %s. Exiting...\n", strerror(errno)); exit(EXIT_FAILURE); }

	fprintf(output, "%s", "\t\tUnique Networks\n\n");
	fprintf(output, "input : %s\n", inputFile);
	fprintf(output, "class : %s\n", classFile);
	fprintf(output, "geneset : %s\n\n\n", genesetFile);
	int edgePos = 0;

	//loop through all the unique networks
	for (int i = 0; i < networkNum; i++)
	{
		int total;
		if (i == networkNum - 1)
		{
			total = totalEdges - edgesPN[i];
		}
		else
		{
			total = edgesPN[i + 1] - edgesPN[i];
		}
		fprintf(output, "network %d (%d)\t number of edges : %d\n", i + 1, networkIds[i], total);
		fprintf(output, "%s", "---------------");
		fprintf(output, "%s", "---------------\n\n");
		if (total == 0)
		{
			continue;
		}
		//loop through each node in the ith network
		for (int j = 0; j < genesetlength - 1; j++)
		{
			//if change in node values --> edge has been found
			if (nodes[(i * genesetlength) + j + 1] != nodes[(i * genesetlength) + j])
			{
				int change = nodes[(i * genesetlength) + j + 1] - nodes[(i * genesetlength) + j];
				total -= change;
				for (int k = 0; k < change; k++)
				{
					fprintf(output, "%s - %s\n", genesetgenes[j], genesetgenes[edges[edgePos++]]);
					//fprintf(output, "%d - %d ---> normal case total remaining : %d j : %d\n", j, pUniEdges[edgePos++], total, j);
				}
			}
			if (j + 1 == genesetlength - 1 && total > 0)
			{
				//fprintf(output, "%s", "9s case\n");
				for (int k = 0; k < total; k++)
				{
					//fprintf(output, "%d - %d ---> 9s case\n", j + 1, pUniEdges[edgePos++]);
					fprintf(output, "%s - %s\n", genesetgenes[j + 1], genesetgenes[edges[edgePos++]]);
				}
			}



		}
		fprintf(output, "%s", "\n");
		fprintf(output, "%s", "\n");

	}

	fclose(output);
	printf("Finished writing %s\n", outputFile);
}


typedef struct graphNode {
	int node;
	int edge;
	int inBoth;
} graphNode;

__host__ void writeEdgeListFile(char *outputFile, char *inputName, char *className, char *pathwayName, char genesetgenes[][40],
	int genesetlength, int *nodes, int *edges, int *edgesPN, int *priorMatrix, char class1[50], char class2[50])
{
	FILE *output = fopen(outputFile, "w");
	//arrays to keep track of found nodes w/edges for each class	
	std::vector<graphNode> class1Nodes;
	std::vector<graphNode> class2Nodes;


	//loop through *nodes/*edges to find get nodes and edge
	//store found nodes into either class1Nodes or class2Nodes with a graphNode struct
	int edgePos = 0;
	char classes[2][50];
	strcpy(classes[0], class1);
	strcpy(classes[1], class2);
	//loop through all the unique networks
	for (int i = 0; i < 2; i++)
	{
		int total;
		if (i == 2 - 1)
		{
			total = edgesPN[2] - edgesPN[i];
		}
		else
		{
			total = edgesPN[i + 1] - edgesPN[i];
		}
		if (total == 0)
		{
			continue;
		}
		//loop through each node in the ith network
		for (int j = 0; j < genesetlength - 1; j++)
		{
			//if change in node values --> edge has been found
			if (nodes[(i * genesetlength) + j + 1] != nodes[(i * genesetlength) + j])
			{
				int change = nodes[(i * genesetlength) + j + 1] - nodes[(i * genesetlength) + j];
				total -= change;
				for (int k = 0; k < change; k++)
				{
					graphNode temp;
                                        temp.node = j;
                                        temp.edge = edges[edgePos++];
					temp.inBoth = 0;
					//fprintf(output, "%s - %s\n", genesetgenes[j], genesetgenes[edges[edgePos++]]);
					//printf("%s\t%s\t%s\n", genesetgenes[j], genesetgenes[edges[edgePos++]], classes[i]);
					//printf("%s\n", classes[i]);
					if(strcmp(classes[i], class1) == 0)
					{
						class1Nodes.push_back(temp);
					}
					else
					{
						class2Nodes.push_back(temp);
					}
					//fprintf(output, "%d - %d ---> normal case total remaining : %d j : %d\n", j, pUniEdges[edgePos++], total, j);
				}
			}
			if (j + 1 == genesetlength - 1 && total > 0)
			{
				for (int k = 0; k < total; k++)
				{
					graphNode temp;
                                        temp.node = j + 1;
                                        temp.edge = edges[edgePos++];
					temp.inBoth = 0;
					//fprintf(output, "%d - %d ---> 9s case\n", j + 1, pUniEdges[edgePos++]);
					//fprintf(output, "%s - %s\n", genesetgenes[j + 1], genesetgenes[edges[edgePos++]]);
					//printf("%s\t%s\t%s\n", genesetgenes[j + 1], genesetgenes[edges[edgePos++]], classes[i]);
					//printf("%s\n", classes[i]);
                                        if(strcmp(classes[i], class1) == 0)
                                        {
                                                class1Nodes.push_back(temp);
                                        }
                                        else
                                        {
                                                class2Nodes.push_back(temp);
                                        }
				}
			}
		}
	}


	//determine which nodes in class1 are also in class2 - if the are mark inBoth for each node so later we dont repeat when we print
	for(int i = 0; i < class1Nodes.size(); i++)
	{
		for(int j = 0; j < class2Nodes.size(); j++)
		{
			if(class1Nodes.at(i).node == class2Nodes.at(j).node && class1Nodes.at(i).edge == class2Nodes.at(j).edge)
			{
				class1Nodes.at(i).inBoth = 1;
				class2Nodes.at(j).inBoth = 1;
				
			}
		}
	}

	//create spacer and ff rows to offset our reads for row/col access to the prior matrix
 	int c = ((genesetlength * genesetlength) - genesetlength) / 2;
	int position = 0;
        int *spacer = (int *)malloc(sizeof(int) * c);
        int *ff = (int *)malloc(sizeof(int) * c);
        for (int row = 1; row < genesetlength; row++)                     
        {
                for (int col = 0; col < row; col++)
                {
                        spacer[position] = row;
                        ff[position] = col;
                        position++;
                }
        }

	//denote if eddy found an edge so that we can check later if the edge wasnt found- 1 = EDDY found edge 0 = EDDY  did not find edge
	int *eddyFound = (int *)calloc(genesetlength * genesetlength, sizeof(int));
	

	//printf("Print contents of class1: \n");
	//for(int i = 0; i < class1Nodes.size(); i++)
	//{
	//	printf("%s\t%s\t in Both: %d\n", genesetgenes[class1Nodes.at(i).node], genesetgenes[class1Nodes.at(i).edge], class1Nodes.at(i).inBoth);
	//}
	//printf("\nPrint contents of class2 : \n");
	//for(int i = 0; i < class2Nodes.size(); i++)
	//{
	//	printf("%s\t%s\t in Both: %d\n", genesetgenes[class2Nodes.at(i).node], genesetgenes[class2Nodes.at(i).edge], class2Nodes.at(i).inBoth);
	//}


	
	char priorString[2][20] = {"NONE", "PRIOR"};
	//printf("ATTEMPTING FILE OUTPUT TEST: \n");
	for(int i = 0; i < class1Nodes.size(); i++)
	{
		
		int row = class1Nodes.at(i).node;
                int col = class1Nodes.at(i).edge;
		if(class1Nodes.at(i).inBoth == 1)
		{
			//if node is in both classes print to file with "Both" class specifier
			fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], "Both", priorString[*(priorMatrix + row * genesetlength + col)]);
			//printf("%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], "Both", priorString[*(priorMatrix + row * genesetlength + col)]);
			//denote that eddy found this edge
			*(eddyFound + row * genesetlength + col) = 1;
			*(eddyFound + col * genesetlength + row) = 1;
			//printf("IN BOTH!");
		}
		else
		{
			//if it isnt both class write all class1 nodes
			fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], class1, priorString[*(priorMatrix + row * genesetlength + col)]);
			//denote that eddy found this edge
			*(eddyFound + row * genesetlength + col) = 1;
                        *(eddyFound + col * genesetlength + row) = 1;
		}
	}
	
	for(int i = 0; i < class2Nodes.size(); i++)
	{
		int row = class2Nodes.at(i).node;
                int col = class2Nodes.at(i).edge;
		if(class2Nodes.at(i).inBoth != 1)
		{
			//write to file all nodes as long as not also in class1 - that way we dont repeat edges
			fprintf(output, "%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], class2, priorString[*(priorMatrix + row * genesetlength + col)]);
			//printf("%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], class2, priorString[*(priorMatrix + row * genesetlength + col)]);
			//denote that eddy found an edge
			*(eddyFound + row * genesetlength + col) = 1;
                        *(eddyFound + col * genesetlength + row) = 1;
		}
	}
	

	//write edges that werent found by EDDY
	//loop through prior matrix- if relationship isnt already found by eddy but in the prior file, write relationship to file
	for(int i = 0; i < c; i++)
	{
		int row = spacer[i];
		int col = ff[i];
		//0 in eddyFound = eddy did not find 1 in eddyFound = eddy did find
		if(*(eddyFound + row * genesetlength + col) == 0 && *(priorMatrix + row * genesetlength + col) == 1)
		{
			fprintf(output,"%s\t%s\t%s\t%s\n", genesetgenes[row], genesetgenes[col], "Neither", "PRIOR");
		}
	}
	
	free(spacer); spacer = NULL;
	free(ff); ff = NULL;
	free(eddyFound); eddyFound = NULL;
	
	fclose(output);
	printf("Finished writing edgelist file %s\n", outputFile);

}

__host__ double mean(double *a, size_t size)
{
	double sum = 0.0;
	for (size_t i = 0; i < size; i++)	{ sum += a[i];	}
	return sum / size;
}

__host__ double variance(double mean, double *a, size_t size)
{
	double sum = 0.0;
	for (size_t i = 0; i < size; i++)	{ sum += pow(a[i] - mean, 2); }
	return sum / size;
}

//__host__ double betaCDF(double x, double alpha, double beta)
//{
//	printf("Beta function started!\n");
//	return Incomplete_Beta_Function(x, alpha, beta) / Beta_Function(alpha, beta);
//}

__host__ void checkParentLimit(int numNetworks, int numNodes, int maxParents, int *nodes, size_t size)
{
	//loop through each network
	for (int net = 0; net < numNetworks; net++)
	{
		//loop through nodes in each network
		for (int i = 0; i < numNodes - 1; i++)
		{
			if (nodes[(net * numNodes) + i + 1] - nodes[(net * numNodes) + i] > maxParents)
			{
				printf("PROBLEM @ %d %d\n", (net * numNodes) + i, (net * numNodes) + i + 1);
			}
		}
	}
}


__global__ void noStates_kernel(const int noGenes, int samples1, int samples2, int *data1, int *data2, int *out1, int *out2)
{
int *dataIn;
	int samplesIn;
	int start;
	int stop;
	int retVal;


	if (blockIdx.x < noGenes){
		dataIn = data1;
		samplesIn = samples1;
		start = blockIdx.x*samplesIn;
		stop = start + samplesIn;
		assert(stop <= noGenes * samples1);
	}
	else{
		dataIn = data2;
		samplesIn = samples2;
		start = (blockIdx.x - noGenes)*samplesIn;
		stop = start + samplesIn;
		assert(stop <= noGenes * samples2);
	}



	unsigned short statedata1[3] = { 0, 0, 0 };

	for (int i = start; i < stop; i++)
	{
	
		if (dataIn[i] == -1){
			statedata1[0]++;
		}
		if (dataIn[i] == 0){
			statedata1[1]++;

		}
		if (dataIn[i] == 1){
			statedata1[2]++;

		}

	}


	retVal = 3;
	for (int i = 0; i < 3; i++){
		out2[blockIdx.x * 3 + i] = 1;
		if (statedata1[i] == 0){
			out2[blockIdx.x * 3 + i] = 0;

			retVal--;
		}




	}
	
	out1[blockIdx.x] = retVal;
}

int main(int argc, char *argv[])
{
	//looking at GPU properties
	int nDevices;
	//int maxBlocks;
	int maxThreads;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Processor Clock Rate (KHz): %d\n", prop.clockRate);
		printf("  Device Max Number of Blocks: %d\n",
			prop.maxGridSize[1]);
		
		printf("  Device Max Number of Threads per Block: %d\n",
			prop.maxThreadsPerBlock);
		maxThreads = prop.maxThreadsPerBlock;
		printf("  Device Max Number of Compute Indices: %d\n",
			prop.maxGridSize[1] * prop.maxThreadsPerBlock);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Compute Capability : %d.%d\n", prop.major, prop.minor);
		printf("  Device has %d SMs\n", prop.multiProcessorCount);
		printf("  This device can run multiple kernels simultaneously : %d \n\n",
			prop.concurrentKernels);
		

	}

	int startT = getMilliCount();
	int start1 = getMilliCount();
	//data grab routine *************************************************************************************
	//command line arguments parser--------------------------------------------------------------------------

	char *inputFile = NULL;
	char *classFile = NULL;
	char *genesetFile = NULL;
	//limits the number of parents a node can have
	int parentCap = 0;
	//number of permutations
	int perms = 0; // 1 permutation
	double pw = 1.0; //default - no prior weight
	double pThreshold = .05; //default value

	//-d for input
	//-g for geneset
	//-c for class
	//-mp for max parents
	//-p for p threshold value
	//-r number of permutations
	//-pw prior weight = [0,1]
	//loop through argv, determining location of each arg parameter
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-d") == 0)
			inputFile = argv[i + 1];
		else if (strcmp(argv[i], "-g") == 0)
			genesetFile = argv[i + 1];
		else if (strcmp(argv[i], "-c") == 0)
			classFile = argv[i + 1];
		else if (strcmp(argv[i], "-mp") == 0)
			parentCap = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0)
			printf("Required arguments : \n -d input.txt\n -c classfile.txt\n -g geneset.txt -mp # of parents\n -p pvalue for independence testing\n");
		else if (strcmp(argv[i], "-r") == 0)
			perms = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "-pw") == 0)
			pw = atof(argv[i + 1]);
		else if(strcmp(argv[i], "-p") == 0)
			pThreshold = atof(argv[i+1]);
	}

	
	
	
	
	//set to defaults if no arguments are included
	if (inputFile == NULL)	{ printf("Invalid input file entered. Exiting...\n"); exit(EXIT_FAILURE); }
	if (classFile == NULL)	{ printf("Invalid class file entered. Exiting...\n"); exit(EXIT_FAILURE); }
	if (genesetFile == NULL)	{ printf("Invalid geneset file entered. Exiting...\n"); exit(EXIT_FAILURE); }
	printf("%s\n", genesetFile);
	if(pThreshold < 0.0 || pThreshold > 1.0)
	{
		pThreshold = .05; //default if out of range
	}
	printf("p Threshold  = %f\n", pThreshold); 
	//set maxparents to default 3 if not set in command line arguments
	if (parentCap <= 0)
	{
		//should normally be run with 3 which will make it for a total of 4
		parentCap = 3;
	}
	printf("Max parents = %d\n", parentCap);
	const int MAX_PARENTS = parentCap;
	if (perms <= 0)
	{
		perms = 100; //default
	}
	printf("Permutations = %d\n", perms);
	if (pw < 0.0 || pw > 1.0)
	{
		pw = 1.0; //no prior knowledge
	}
	printf("pw : %f\n", pw);
	//end command line parser---------------------------------------------------------------------------------------

	//expression data
	FILE *fp = fopen(inputFile, "r");
	
	//class 
	FILE *fp2 = fopen(classFile, "r");

	//FILE * fp3 = fopen("geneset10.txt", "r");
	FILE *fp3 = fopen(genesetFile, "r");

	FILE *results = fopen("results.txt", "w");
	

	//check that files are working
	if (fp == NULL)	{ printf("Expression file is NULL. error number is : %d\n", strerror(errno)); exit(EXIT_FAILURE); }
	if (fp2 == NULL){ printf("Class File File is NULL. error number is : %d\n", strerror(errno)); exit(EXIT_FAILURE); }
	if (fp3 == NULL){ printf("Gene List File is NULL. error number is : %d\n", strerror(errno)); exit(EXIT_FAILURE); }
	if (results == NULL){ printf("results file is NULL. error number is : %d\n", strerror(errno)); exit(EXIT_FAILURE); }
	

	//allocate memory for file reads
	char buf[200000];
	int numsamples = 0;
	int numgenes = 0, genesetlength = 0;
	int numclass1, numclass2;
	char sampnames[5000][50];
	char classnames[5000][50];
	char genesetgenes[100][40];
	int classids[5000];
	char class1[50], class2[50];
	char genenames[20500][16]; //used to be [20000][10]- not big enough --> changed stack size to 1.5MB
	int genesetindexintodata[50];
	//int classindexintodata[100][2];
	int i, j;// , index;
	//int jindex1, jindex2;
	int *data;
	int *transferdata1;
	int *transferdata2;
	char *token;


	//loads expression file into buffer
	fgets(buf, sizeof(buf), fp);
	token = strtok(buf, "\t");
	token = strtok(NULL, "\t");

	// Skip first word "Genelist"
	//load  samplenames from buffer and count number of samples
	while (token != NULL) {
		strcpy(sampnames[numsamples], token);
		numsamples++;
		token = strtok(NULL, "\t");
	}

	for (int i = 0; i < numsamples; i++) {
		// Get rid of extra empty "sample" caused by trailing tab
		if (strlen(sampnames[i]) == 1) numsamples--;
	}

	printf("%d samples\n", numsamples);

	while (fgets(buf, sizeof(buf), fp)) {
		token = strtok(buf, "\t");
		strcpy(genenames[numgenes], token);
		//    printf("%s\n", genenames[numgenes]);
		numgenes++;
	}

	numgenes--;
	printf("%d genes\n", numgenes);

	data = (int *)malloc(numgenes*numsamples*sizeof(int));
	//reset file position to 0
	fseek(fp, 0, 0);
	// Skip first line
	fgets(buf, sizeof(buf), fp);
	for (i = 0; i < numgenes; i++) {
		fgets(buf, sizeof(buf), fp);
		token = strtok(buf, "\t");
		for (j = 0; j < numsamples; j++) {
			token = strtok(NULL, "\t");
			assert(i * numsamples + j < numgenes * numsamples);
			sscanf(token, "%d", &data[i*numsamples + j]);
		}
	}

	fclose(fp);
	fgets(buf, sizeof(buf), fp2);
	token = strtok(buf, "\t");
	for (i = 0; i < numsamples; i++) {
		strcpy(classnames[i], token);
		token = strtok(NULL, "\t");
	}
	for (i = 1; i < numsamples; i++) {
		if (strcmp(classnames[i], classnames[0])) break;
	}
	strcpy(class1, classnames[0]);
	strcpy(class2, classnames[i]);
	numclass1 = 0;
	numclass2 = 0;

	//check if classfile had newline character at end of final classname, preventing 
	//strcmp from working
	if (classnames[numsamples - 1][strlen(classnames[numsamples - 1]) - 1] == '\n')
	{
		classnames[numsamples - 1][strlen(classnames[numsamples - 1]) - 1] = '\0';
	}

	for (i = 0; i < numsamples; i++) {
		if (!strcmp(classnames[i], class1)) {
			numclass1++;
			classids[i] = 0;

		}
		if (!strcmp(classnames[i], class2)) {
			numclass2++;
			classids[i] = 1;
		}

	}

	//should this be done? when reading in CTRP sample data not all classids are filled
	//this leads to a problem when reshuffling them later - only finding 201/202 supposed samples
	/*int revisedSamples = 0;
	for (int i = 0; i < numsamples; i++)
	{
		if (classids[i] == 1 || classids[i] == 0)
		{
			revisedSamples++;
		}
	}
	printf("revised : %d original : %d\n", revisedSamples, numsamples);
	numsamples = revisedSamples;*/

	//  printf("\n");
	printf("Classes: %d %s, %d %s\n", numclass1, class1, numclass2, class2);
	fclose(fp2);
	clock_t cpuTime = clock(), diff;
	//-----------------------------------------------------------------------------------
	//-----------------------------------------------------------------------------------

	fprintf(results, "%s\t %s\t %s\t\n", class2, "JS", "P");
	while (fgets(buf, sizeof(buf), fp3))
	{
		//check if beginning of file is newline
		if (buf[0] == '\n')
		{
			continue;
		}
		//ensures that classids are restored to a preshuffled state
		//for the first run of each pathway before permutations begin
		for (int k = 0; k < numsamples; k++)
		{
			if (strcmp(classnames[k], class1) == 0)
			{
				classids[k] = 0;
			}
			if (strcmp(classnames[k], class2) == 0)
			{
				classids[k] = 1;
			}

		}

		char *pathwayName = strtok(buf, "\n");
		//fgets(buf, sizeof(buf), fp3);
		//token = strtok(buf, "\t");
		token = strtok(pathwayName, "\t");
		token = strtok(NULL, "\t");
		// Get first word "Geneset"
		token = strtok(NULL, "\t");
		// Skip second word "URL"
		genesetlength = 0;
		while (token != NULL) {
			strcpy(genesetgenes[genesetlength], token);
			genesetlength++;
			token = strtok(NULL, "\t");
		}

		printf("%s\n", pathwayName);
		
		//-------------------------------------------------------------------------
		// Get rid of trailing carriage return on last gene name
		//no longer needed because strok with pathwayName eliminates newline character
		//genesetgenes[genesetlength - 1][strlen(genesetgenes[genesetlength - 1]) - 1] = '\0';

		printf("%d genes in geneset\n", genesetlength);
		//fclose(fp3);

		//should this be moved to after adjusting genesetlength?
		//transferdata1 = (int *)malloc(genesetlength*numclass1*sizeof(int));
		//transferdata2 = (int *)malloc(genesetlength*numclass2*sizeof(int));
		
		//accounts for any missing/extra genes
		int indexPos = 0;
		for (i = 0; i < genesetlength; i++) {
			int flagFound = 0;
			genesetindexintodata[i] = -1;
			for (j = 0; j < numgenes; j++) {
				if (!strcmp(genenames[j], genesetgenes[i])) {
					flagFound = 1;
					//fill genesetgenes only with genes that are being evaluated
					strcpy(genesetgenes[indexPos], genenames[j]);
					genesetindexintodata[indexPos] = j;
					break;
				}
			}
			//	printf ("Gene %d index: %d %d %s %s\n",i,genesetindexintodata[i],j,genenames[genesetindexintodata[i]],genesetgenes[i]);
			if (flagFound)
			{
				indexPos++;
			}
		}
		transferdata1 = (int *)malloc(genesetlength*numclass1*sizeof(int));
		transferdata2 = (int *)malloc(genesetlength*numclass2*sizeof(int));
		//delete genes that shouldn't be in gene list
		for (int k = indexPos; k < genesetlength; k++)	{ genesetgenes[k][0] = '\0'; }
		//adjust # of genes
		genesetlength = indexPos;
		printf("Adjusted genes : %d\n", genesetlength);
		
		
		//prior knowledge load data into binary matrix-------------------------------------------------------------
		
		int *priorMatrix = (int *)calloc(genesetlength * genesetlength, sizeof(int)); //array to hold prior knowledge matrix

		//look into \PRIORS folder
		char directory[300];; //directory for prior files
		strcpy(directory, DIR); //load folder path depending on if unix or windows
		char fileName[1000]; //name of prior file
		strcpy(fileName, pathwayName);
		strcat(fileName, ".prior"); //take pathwayname and add .prior to get file path
		strcat(directory, fileName);
		if(strstr(directory, "\r") != NULL){
			printf("File problem! Uses window endings!\n");
		}
		
		FILE *priorFile = fopen(directory, "r"); //open prior knowledge file
		printf("file : %s\n", directory);
		int priorFlag = 1; //1 = files found 0 = no file found
		if (priorFile == NULL)
		{
			printf("No prior file exists. Computing without prior knowledge\n");
			priorFlag = 0;
		}
		//fill prior Matrix
		char priorBuffer[100];
		while (priorFlag && fgets(priorBuffer, sizeof(priorBuffer), priorFile))
		{
			char *tok = strtok(priorBuffer, "\t");
			//printf("gene1 : %s\n", tok);
			int insideFlag = 0;
			int row = -1, col = -1;
			for (int k = 0; k < genesetlength; k++)
			{
				if (strcmp(genesetgenes[k], tok) == 0)
				{
					insideFlag = 1;
					row = k;
					break;
				}
			}

			if (insideFlag == 0)
				continue;
			tok = strtok(NULL, "\t");
			//printf("relationship : %s\n", tok);
			if (strcmp(tok, "neighbor-of") == 0)
			{
				continue;
			}
			tok = strtok(NULL, "\t");
			tok[strlen(tok) - 1] = '\0';
			//printf("gene2 : %s\n", tok);
			insideFlag = 0;
			for (int k = 0; k < genesetlength; k++)
			{
				if (strcmp(genesetgenes[k], tok) == 0)
				{
					insideFlag = 1;
					col = k;
					break;
				}
			}
			if (insideFlag == 0)
				continue;
			assert(row > -1 && col > -1 && row < genesetlength && col < genesetlength);
			*(priorMatrix + row * genesetlength + col) = 1;
			*(priorMatrix + col * genesetlength + row) = 1;
		}

		if (priorFlag == 1) //only try closing file if it was open to begin with
		{
			fclose(priorFile);
		}
			
		//begin permutation loop
		int n;
		int x;

		//stores js values across permutations for p value calcs
		double *jsVals = (double *)malloc(sizeof(double) * perms);
		//int *triAry2;
		//int *triAry3;
		//used to print network/bdeu score files
		int first_unisum;
		int first_scaler;
		int *first_uniNodes;
		int *first_uniEdges;
		int *first_uniEpn;
		double *first_lval1 = NULL;
		int first_numEdges;
		int *uniqueNetIds = NULL;
		//used to calculate edgeList without parent limit after permuatations finished - stores 1st permutation data
		int *edgeListData1 = NULL;
		int *edgeListData2 = NULL;
		//int *initialSpacr = NULL;
		//int *initialFF = NULL;
		int *initialSearcher = NULL;
		//number range of random numbers needed [0,mems)
		int mems = numsamples;

		float totalTime;
		cudaEvent_t begin, end;
		cudaEventCreate(&begin);
		cudaEventCreate(&end);

		for (int permNum = 0; permNum < perms; permNum++)
		{
			n = 0;
			
			int *randNums = (int *)malloc(sizeof(int) * numsamples);
			for (int c = 0; c < mems; c++) {
				randNums[c] = rand() % mems;
			}
			while (n < mems) {
				int r = rand() % mems;

				for (x = 0; x < n; x++)
				{
					if (randNums[x] == r){
						break;
					}
				}
				if (x == n){
					randNums[n++] = r;
				}
			}

			

			//after first permutation scramble samplings
			if (permNum > 0)
			{
				for (int counter = 0; counter < numclass1; counter++)
				{
					assert(counter < numsamples);
					classids[randNums[counter]] = 0;
				}
				for (int counter = numclass1; counter < numsamples; counter++)
				{
					assert(counter < numsamples);
					classids[randNums[counter]] = 1;
				}
			}
				
			
			
			free(randNums); randNums = NULL;
			
			//sort data into class1 and class 2
			int index = 0;
			for (i = 0; i < genesetlength; i++) {
				if (genesetindexintodata[i] == -1) {
					i++;
				}

				int jindex1 = 0;
				int jindex2 = 0;
				for (j = 0; j < numsamples; j++) {
					/*if (j == 4268 || j = 4 || classids[j] == -858993460)
					{
					printf("bad value in classids accessed @ %d with value of %d\n", i, classids[j]);
					}*/
					if (classids[j] == 0) {
						assert(index * numclass1 + jindex1 < numclass1 * genesetlength);//transferdata
						assert(genesetindexintodata[i] * numsamples + j < numgenes * numsamples); //data
						transferdata1[index*numclass1 + jindex1] = data[genesetindexintodata[i] * numsamples + j];
						jindex1++;
						//			printf("%d ",transferdata1[index*numclass1+jindex1]);
					}
					if (classids[j] == 1) {
						assert(index*numclass2 + jindex2 < numclass2 * genesetlength);//transferdata
						assert(genesetindexintodata[i] * numsamples + j < numgenes * numsamples); //data
						transferdata2[index*numclass2 + jindex2] = data[genesetindexintodata[i] * numsamples + j];
						jindex2++;
					}
				}
				//	printf("\n");
				index++;
			}

			//dead code never run
			//while (fgets(buf, sizeof(buf), fp)) {
			//	token = strtok(buf, "\t");
			//	strcpy(genenames[numgenes], token);
			//	//    printf("%s\n", genenames[numgenes]);
			//	numgenes++;
			//}
			//printf("Data Grab Done\n");
			int genes = genesetlength;
			//end data grab routine ******************************************************************************
			//samples sizes in both classes


			//start timing- Tomas
			cudaEventRecord(begin, 0);



			int samples = numclass1;
			int samples2 = numclass2;

			int c = (((genes*genes) - genes) / 2);

			int scaler = (samples + 1);
			int scaler2 = (samples2 + 1);
			int scalerSum = scaler + scaler2;
			
			//int pos = 0;
			//int posrr = 0;
			//transfer data for processing
			//triAry2 = transferdata1;
			//triAry3 = transferdata2;
			//int arypos = 0;
			//int hold = 0;
			//int spacer = 0;
			//int len = genes;
			
			int* spacer1;//dyn
			spacer1 = (int *)malloc(c*sizeof(int));
			int* ff1;//dyn
			int *searcher;
			ff1 = (int *)malloc(c*sizeof(int));
			searcher = (int *)malloc(genes*sizeof(int));


			//following block of code determines diaganol representation of data matrix
			//searcher[0] = 0;
			//int diff = 0;
			//for (spacer = 0; spacer <= len + 1; spacer++)
			//{
			//	hold = len*spacer;
			//	for (int f = spacer + 1; f < len; f++)
			//	{
			//		spacer1[arypos] = f;
			//		ff1[arypos] = spacer;
			//		arypos++;
			//	}
			//	if (spacer > 0 && spacer < len){
			//		//diff = genes - spacer;
			//		diff = spacer;
			//		printf("diff : %d\n", diff);
			//		searcher[spacer] = searcher[spacer - 1] + diff;
			//	}
			//}
			//for (int i = 0; i < genes; i++)
			//{
			//	printf("searcher[%d] : %d\n", i, searcher[i]);
			//}

			//determines diaganol representation of data matrix
			searcher[0] = 0;
			int position = 0;
			for (int row = 1; row < genes; row++)
			{
				for (int col = 0; col < row; col++)
				{
					assert(position < c);
					spacer1[position] = row;
					//printf("spacer1[%d] : %d\n", position, spacer1[position]);
					ff1[position] = col;
					position++;
					
					
				}
				if (row > 0)
				{
					assert(row < genes);
					searcher[row] = searcher[row - 1] + row;
				}
			}
			
			
			
			//if first permutation store spacr, ff, searcher to use in edgeList calcs after permutations
			if (permNum == 0)
			{
				//initialSpacr = (int *)malloc(sizeof(int) * c);
				//initialFF = (int *)malloc(sizeof(int) * c);
				initialSearcher = (int *)malloc(sizeof(int) * genes);
				//memcpy(initialSpacr, spacer1, sizeof(int) * c);
				//memcpy(initialFF, ff1, sizeof(int) * c);
				memcpy(initialSearcher, searcher, sizeof(int) * genes);
			}

			//start cuda time
			cudaEvent_t start, stop;
			float time;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);


			///cuda launch 1***************************************************************************
			//holds edge data in binary format
			//int onesSize = sizeof(double) * c * scalerSum;
			int *edgesPN;
			edgesPN = (int *)malloc(sizeof(int)* (scalerSum + 1));

			//device copies for out23 and edgesPN
			int *dout23;
			int *dedgesPN;

			//device copies
			int *dtriA, *ddofout, *dtriAb, *dppn, *dstf;
			int *dff, *dspacr;
			//double *d_ones;
			int *dpriorMatrix;
			
			//mem sizes required
			int size2 = c*((samples2 + 1) + (samples + 1))*sizeof(int);
			//int size3 = c*((samples2 + 1) + (samples + 1))*sizeof(double);
			int dppnLength = genesetlength * 2;
			////space alloc for device
			HANDLE_ERROR(cudaMalloc((void **)&dtriA, genesetlength*samples*sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void **)&dtriAb, genesetlength*samples2*sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void **)&dppn, genesetlength * 2 * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void **)&dstf, genesetlength * 2 * 3 * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void **)&ddofout, size2));
			HANDLE_ERROR(cudaMalloc((void **)&dff, c*sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void **)&dspacr, c*sizeof(int)));
			//cudaMalloc((void **)&d_ones, onesSize);
			HANDLE_ERROR(cudaMalloc((void **)&dout23, sizeof(int) * c * scalerSum));
			HANDLE_ERROR(cudaMalloc((void **)&dedgesPN, sizeof(int) * (scalerSum + 1)));

			HANDLE_ERROR(cudaMalloc((void **)&dpriorMatrix, sizeof(int) * genesetlength * genesetlength));
			

			//copy into device 
			assert(genes*samples*sizeof(int) == genesetlength * numclass1 * sizeof(int));
			assert(genes*samples2*sizeof(int) == genesetlength * numclass2 * sizeof(int));
			HANDLE_ERROR(cudaMemcpy(dtriA, transferdata1, genes*samples*sizeof(int), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dtriAb, transferdata2, genes*samples2*sizeof(int), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dff, ff1, c*sizeof(int), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dspacr, spacer1, c*sizeof(int), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dpriorMatrix, priorMatrix, genesetlength * genesetlength * sizeof(int), cudaMemcpyHostToDevice));
			
			

			//no longer used once copied to GPU
			free(spacer1); spacer1 = NULL;
			free(ff1); ff1 = NULL;

			//deploy
			int milliSecondsElapsed1 = getMilliSpan(start1);
			int start2 = getMilliCount();
			int sampleSum = samples + samples2 + 2;
			//printf("samples : %d\n", samples);

			
			
			//run no states in separate kernel to avoid threading
			//noStates_kernel <<<genes * 2, 1 >>>(genes, samples, samples2, dtriA, dtriAb, dppn, dstf);





			cudaEventRecord(start, 0);
			//printf("c = %d\n", c);
			if( c < MAX_THREADS)
			{
			run2 << <sampleSum, c, genes * genes * sizeof(int) >> >(genes, samples, samples2, dtriA, dtriAb, dspacr, dff, ddofout, dppn, dstf, dout23, c, dpriorMatrix, pw);
			}
			else
			{
				int BPN = ceil((c * 1.0) / MAX_THREADS);
				int TPB = ceil((c * 1.0) / BPN);
			
				//printf("launching with %d blocks per network and %d threads per block\n", BPN, TPB);
				run2Scalable <<< sampleSum * BPN, TPB>>>(genes, samples, samples2, dtriA, dtriAb,dspacr, dff, ddofout, dppn, dstf, dout23, c, dpriorMatrix, pw, BPN, TPB);
				//printf("run2Scalable completed\n");
			}


			//test ppn/stf
			/*int *tempPpn = (int *)malloc(sizeof(int) * 2 * genesetlength);
			int *tempStf = (int *)malloc(sizeof(int) * 2 * 3 * genesetlength);
			cudaMemcpy(tempPpn, dppn, sizeof(int) * 2 * genesetlength, cudaMemcpyDeviceToHost);
			cudaMemcpy(tempStf, dstf, sizeof(int) * 2 * 3 * genesetlength, cudaMemcpyDeviceToHost);
			for (int i = 0; i < 2 * genesetlength; i++)
			{
				printf("ppn[%d] : %d\n", i, tempPpn[i]);
			}
			for (int i = 0; i < 2 * 3 * genesetlength; i++)
			{
				printf("stf[%d] : %d\n", i, tempStf[i]);
			}*/

			//printf("run2 finished\n");
			cudaError_t errSync = cudaGetLastError();
			if (errSync != cudaSuccess)
			{
				printf("%s\n", cudaGetErrorString(errSync));
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			//printf("Run 2 Time : %f\n", time);

			if (permNum == 0)
			{
				//holds post run2 edge data for edge list calculations after permutations
				edgeListData1 = (int *)malloc(sizeof(int) * c);
				edgeListData2 = (int *)malloc(sizeof(int) * c);

				//host array to transfer output of run2 to edgeListData1/edgeListData2
				int *tempOut23 = (int *)malloc(sizeof(int) * c * scalerSum);
				

				//copy binary data back to CPU
				HANDLE_ERROR(cudaMemcpy(tempOut23, dout23, sizeof(int) * c * scalerSum, cudaMemcpyDeviceToHost));
				

				//first network in first class - no samples left out
				for (int i = 0; i < c; i++)
				{
					//load 1st network from class 1
					edgeListData1[i] = tempOut23[i];
				}
				int count = 0;
				//last network in 2nd class - no samples left out
				for (int i = (scalerSum - 1) * c; i < (scalerSum) * c; i++)
				{
					edgeListData2[count++] = tempOut23[i];
				}
			


				////copy data for the first network in the first class
				//int *ptr1 = &tempOut23[0];
				//memcpy(edgeListData1, ptr1, sizeof(int) * c);
				////copy data for the first network in the second class
				//printf("2nd memcpy starting point : %d\n", scaler * c);
				////int *ptr2 = &tempOut23[scaler * c];
				//int *ptr2 = &tempOut23[(scaler) * c];
				//memcpy(edgeListData2, ptr2, sizeof(int) * c);
				//ptr1 = NULL;
				//ptr2 = NULL;

free(tempOut23); tempOut23 = NULL;
			}



			int milliSecondsElapsed2 = getMilliSpan(start2);
			int start3 = getMilliCount();



			//device copy
			int *dsrchAry, *tempEdgesSums;
			HANDLE_ERROR(cudaMalloc((void **)&dsrchAry, genes * sizeof(int)));
			HANDLE_ERROR(cudaMemcpy(dsrchAry, searcher, genes * sizeof(int), cudaMemcpyHostToDevice));
			tempEdgesSums = (int *)calloc(sampleSum + 1, sizeof(int));

			free(searcher); searcher = NULL;

			cudaEvent_t PN_start, PN_stop;
			cudaEventCreate(&PN_start);
			cudaEventCreate(&PN_stop);
			cudaEventRecord(PN_start, 0);
			float PN_time;

			//edgePerNetworkKernel << < sampleSum + 1, c, (c * sizeof(int)) >> >(dout23, dedgesPN, dsrchAry, genes, MAX_PARENTS, c);
			edgePerNetworkKernel << < sampleSum + 1, 1 >> > (dout23, dedgesPN, dsrchAry, genes, MAX_PARENTS, c);
			//printf("edgesPerNetworkKernel finished\n");
			cudaEventRecord(PN_stop, 0);
			//HANDLE_ERROR(cudaMemcpy(edgesPN, dedgesPN, sizeof(int) * (scalerSum + 1), cudaMemcpyDeviceToHost));
			
			//copy edge sums over to CPU to calculate prefix sum for edgesPN	
			HANDLE_ERROR(cudaMemcpy(tempEdgesSums, dedgesPN, sizeof(int) * (scalerSum + 1), cudaMemcpyDeviceToHost));	
			
			edgesPN[0] = 0;
			for(int i = 1; i < sampleSum + 1; i++)
			{
				edgesPN[i] = edgesPN[i-1] + tempEdgesSums[i-1]; //prefix sum calculation
			}
			//get rid of this temp array	
			free(tempEdgesSums); tempEdgesSums = NULL;
			//edgesPN on the CPU is now fixed but dedgesPN is used later- copy edgesPN results back to GPU memory
			HANDLE_ERROR(cudaMemcpy(dedgesPN, edgesPN, sizeof(int) * (sampleSum + 1), cudaMemcpyHostToDevice));
			
			/*
			for (int i = 0; i < scalerSum + 1; i++)
			{
			printf("edgesPN[%d] : %d\n", i, edgesPN[i]);
			}	
			*/
			//exit(EXIT_FAILURE);
			
			errSync = cudaGetLastError();
			if (errSync != cudaSuccess)
			{
				printf("%s\n", cudaGetErrorString(errSync));
			}

			/*for (int i = 0; i < scalerSum + 1; i++)
			{
			printf("edgesPN[%d] : %d\n", i, edgesPN[i]);
			}*/
			//cudaEventRecord(PN_stop, 0);
			cudaEventSynchronize(PN_stop);
			cudaEventElapsedTime(&PN_time, PN_start, PN_stop);
			//printf("edgesPerNetworkKernel time : %f\n", PN_time);
			//cudaFree(d_ones); 
			HANDLE_ERROR(cudaFree(dpriorMatrix)); dpriorMatrix = NULL;
			HANDLE_ERROR(cudaFree(ddofout)); ddofout = NULL;
			HANDLE_ERROR(cudaFree(dff)); dff = NULL;
			HANDLE_ERROR(cudaFree(dspacr)); dspacr = NULL;//cudaFree(dtriA); cudaFree(dtriAb);-used later in run4
			/***********************************************************************************************************************************************************/
			//total number of edges
			int numEdges = edgesPN[scalerSum];

			//int N = c;
			//int M = genesetlength - 1;
			//int size1 = sizeof(int)*N*(scalerSum);
			//int size222 = sizeof(double)*N*(scalerSum);
			//*****************************************************************************************
			//run22 launch- create parent graphs
			int noNodes = genesetlength;
			//host copies
			int *pNodes, *pEdges;


			//dev copies
			int *dpEdges, *dpNodes;

			//mem reqs
			int nodeSize = sizeof(int)*(noNodes*(scalerSum));
			int edgeSize = sizeof(int)*numEdges;


			//space alloc for device
			HANDLE_ERROR(cudaMalloc((void **)&dpEdges, edgeSize));
			HANDLE_ERROR(cudaMalloc((void **)&dpNodes, nodeSize));


			//space alloc for host
			pNodes = (int *)malloc(nodeSize);
			pEdges = (int *)malloc(edgeSize);
			//FILE *edgePNFile = fopen("edgePN2.txt", "w");
			//for(int i = 0; i < scalerSum + 1; i++)
			//{
			//	fprintf(edgePNFile, "edgesPN[%d] : %d\n", i, edgesPN[i]);
			//}
			//fclose(edgePNFile);
			run22 << <scalerSum, noNodes >> >(c, dedgesPN, dout23, dpNodes, noNodes, numEdges, dsrchAry, dpEdges, MAX_PARENTS);
			//printf("run22 finished\n");

			HANDLE_ERROR(cudaMemcpy(pNodes, dpNodes, nodeSize, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(pEdges, dpEdges, edgeSize, cudaMemcpyDeviceToHost));

			/*for (int i = 0; i < nodeSize / sizeof(int); i++)
			{
				if (i > edgeSize / sizeof(int))
				{
					printf("nodes[%d] : %d\n", i, pNodes[i]);
				}
				else
				{
					printf("nodes[%d] : %d edges[%d] : %d\n", i, pNodes[i], i, pEdges[i]);
				}
			}*/

			/*if (permNum == 0)
			{
				
				for (int i = 0; i < noNodes; i++)
				{
					printf("pNodes[%d] : %d\n", i, pNodes[i]);
				}
				for (int i = 11 * noNodes; i < (11 * noNodes) + noNodes; i++)
				{
					printf("pNodes[%d] : %d\n", i, pNodes[i]);
				}
			}*/
			

			//ensure parent limit
			checkParentLimit(scalerSum, noNodes, MAX_PARENTS, pNodes, nodeSize / sizeof(int));
			/*for (int i = 0; i < nodeSize / sizeof(int); i++)
			{
				if (i > edgeSize / sizeof(int))
				{
					printf("pNodes[%d] : %d\n", i, pNodes[i]);
				}
				else
				{
					printf("pNodes[%d] : %d\t pEdges[%d] : %d\n", i, pNodes[i], i, pEdges[i]);
				}
			}*/
			/*FILE *outputFile = fopen("NodesEdges2.txt", "w");
			for(int i = 0; i < nodeSize / sizeof(int); i++)
			{
				fprintf(outputFile, "pNodes[%d] : %d\n", i, pNodes[i]);
			}
			for(int i = 0; i < edgeSize / sizeof(int); i++)
			{
				fprintf(outputFile, "pEdges[%d] : %d\n", i, pEdges[i]);
			}

			fclose(outputFile); */
			//printf("%d\n", numEdges);
			
			HANDLE_ERROR(cudaFree(dsrchAry)); dsrchAry = NULL;
			HANDLE_ERROR(cudaFree(dout23)); dout23 = NULL;
			//end run 22**********************************************************************************************/

			//start processs to identify unique networks
			int scalerCombo = (scalerSum*scalerSum - scalerSum) / 2;
			//host
			int *scalerTest; //compare value
			int *shrunk;
			int *shrunkPlc; //compare to
			scalerTest = (int *)malloc(sizeof(int)*scalerCombo);
			shrunk = (int *)malloc(sizeof(int)*scalerCombo);
			shrunkPlc = (int *)malloc(sizeof(int)*scalerCombo);


			//see line 132 for more info
			idPrep(scalerSum, scalerCombo, scalerTest, shrunkPlc);

			//dev copies
			//launch for run 25 *****************************************************************************************
			int *dshrunk;
			int *dscalerTest;
			int *dshnkplc;
			HANDLE_ERROR(cudaMalloc((void**)&dshrunk, sizeof(int)*scalerCombo));
			HANDLE_ERROR(cudaMalloc((void**)&dscalerTest, sizeof(int)*scalerCombo));
			HANDLE_ERROR(cudaMalloc((void**)&dshnkplc, sizeof(int)*scalerCombo));

			//cp into device
			HANDLE_ERROR(cudaMemcpy(dscalerTest, scalerTest, sizeof(int)*scalerCombo, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dshnkplc, shrunkPlc, sizeof(int)*scalerCombo, cudaMemcpyHostToDevice));
			//************************************************************************************************
			//(int scaler,int noEdges, int gLength,int scalerCombo, int *dedgesPN, int  *dNodes, int *dedgeAry, int *shrunk)
			// 

			//printf("%/ max: %d   ", maxBlocks*maxThreads);
			//printf("\n");
			run25 << <(scalerCombo / (maxThreads - 1)) + 1, maxThreads - 1 >> >(samples + 1, scalerSum, numEdges, genesetlength, scalerCombo, dedgesPN, dpNodes, dpEdges, dshrunk, dscalerTest, dshnkplc);
			//printf("run25 finished\n");
			//*********************************************************************************************
			HANDLE_ERROR(cudaMemcpy(shrunk, dshrunk, sizeof(int)*scalerCombo, cudaMemcpyDeviceToHost));
			//*************************test****************************************
			cudaFree(dshrunk); dshrunk = NULL;
			cudaFree(dscalerTest); dscalerTest = NULL;
			cudaFree(dshnkplc); dshnkplc = NULL;
			cudaFree(dedgesPN); dedgesPN = NULL;
			cudaFree(dpEdges); dpEdges = NULL;
			cudaFree(dpNodes); dpNodes = NULL;
			free(shrunkPlc); shrunkPlc = NULL;



			bool *uniqueN, *visted;

			//routine for creatation of unique structures 
			uniqueN = (bool *)malloc(sizeof(bool)*scalerSum);

			uniqueN[0] = true;
			visted = (bool *)malloc(sizeof(bool)*scalerSum);
			visted[0] = true;

			
			for (int p = 0; p < scalerSum; p++){
				visted[p] = false;
			}
			for (int p = 0; p < scalerCombo; p++){
				assert(scalerTest[p] < scalerSum);
				if (visted[scalerTest[p]] == true){
					continue;
				}
				else
				{
					if (shrunk[p] == 0){
						uniqueN[scalerTest[p]] = false;
						visted[scalerTest[p]] = true;
					}
					else
					{
						uniqueN[scalerTest[p]] = true;
					}
				}


			}
			//grab network ids from 1st permutation before unique graphs are identified- used when network file is written
			if (permNum == 0)
			{
				uniqueNetIds = (int *)malloc(sizeof(int) * scalerSum);
				int counter = 0;
				for (int i = 0; i < scalerSum; i++)
				{
					if (uniqueN[i])
					{
						uniqueNetIds[counter++] = i;
					}
				}
				uniqueNetIds = (int *)realloc(uniqueNetIds, counter * sizeof(int));
			}


			free(scalerTest); scalerTest = NULL;
			free(shrunk); shrunk = NULL;
			free(visted); visted = NULL;

			int unisum = 0;
			int edSum = 0;
			//should it be scalerSum or scalerSum + 1?
			for (int p = 0; p < scalerSum; p++){

				if (uniqueN[p] == 1){
					unisum++;
					if (p == scalerSum - 1){
						assert(p < scalerSum + 1);
						edSum = edSum + (numEdges - edgesPN[p]);

					}
					else
					{
						assert(p < scalerSum + 1);
						edSum = edSum + edgesPN[p + 1] - edgesPN[p];

					}
				}
			}

			if (permNum == 0)
			{
				printf("Original Number of unique Networks : %d\n", unisum);
			}
			
			//printf("Number of unique networks : %d\n edSum : %d\n numEdges : %d\n edgesPN : %d\n", unisum, edSum, numEdges, edgesPN[scalerSum]);
			//**********************************restructure all **************************************************
			//printf("edSum %d numEdges %d\n", edSum, numEdges);
			int *pUniNodes, *pUniEdges, *pUniEpn;
			//space alloc
			pUniNodes = (int *)malloc(sizeof(int)*unisum*noNodes);
			pUniEdges = (int *)malloc(sizeof(int)*edSum);
			int uniEpnSize = unisum + 1;
			//pUniEpn = (int *)malloc(sizeof(int)*unisum + 1);
			pUniEpn = (int *)malloc(sizeof(int) * uniEpnSize);
			//printf("size of pUniEpn : %d\n", uniEpnSize);


			structureUnique(unisum, numEdges, scaler, scalerSum, noNodes, uniqueN, edgesPN, pEdges, pNodes, pUniEdges, pUniNodes, pUniEpn);
			//printf("structureUnique (NOT A KERNEL) finished\n");
			/*for (int i = 0; i < uniEpnSize; i++)
			{
				printf("pUniEpn[%d] : %d\n", i, pUniEpn[i]);
			}*/
			free(edgesPN); edgesPN = NULL;
			free(pNodes); pNodes = NULL;
			free(pEdges); pEdges = NULL;
			free(uniqueN); uniqueN = NULL;

			//ensure parent limit
			checkParentLimit(unisum, noNodes, MAX_PARENTS, pUniNodes, unisum * noNodes);
			/*for (int i = 0; i < unisum * noNodes; i++)
			{
				if (i > edSum)
				{
					printf("pNodes[%d] : %d\n", i, pUniNodes[i]);
				}
				else
				{
					printf("pNodes[%d] : %d\t pEdges[%d] : %d\n", i, pUniNodes[i], i, pUniEdges[i]);
				}
			}

			for (int i = 0; i < edSum; i++)
			{
				printf("pUniEdges[%d] : %d\n", i, pUniEdges[i]);
			}
			printf("%d\n", edSum);*/




			if (permNum == 0)
			{
				//store graph data for network file write after permutations finished
				//first_uniEpn = (int *)malloc(sizeof(int) * unisum);
				first_uniEpn = (int *)malloc(sizeof(int) * uniEpnSize);
				first_uniNodes = (int *)malloc(sizeof(int) * unisum * noNodes);
				first_uniEdges = (int *)malloc(sizeof(int) * edSum);
				memcpy(first_uniEpn, pUniEpn, unisum * sizeof(int));
				memcpy(first_uniNodes, pUniNodes, unisum * noNodes * sizeof(int));
				memcpy(first_uniEdges, pUniEdges, edSum * sizeof(int));
				first_numEdges = edSum;
				first_unisum = unisum;

			}




			

			scaler = unisum;
			if (permNum == 0)	{ first_scaler = scaler; }
			numEdges = edSum;
			int uniNodeSize = sizeof(int)*(noNodes*unisum);
			int uniEdgeSize = sizeof(int)*numEdges;

			//cuda run 4(final)*************************************************************************************
			double *out5;

			//dev copies
			//int *dtri1; int *dtri2; 
			double *dout5; int *dpEdges2; int *dpNodes2; int *dNij; int *dNijk;
			int *dUniEpn;
			//space alloc dev
			HANDLE_ERROR(cudaMalloc((void **)&dpEdges2, uniEdgeSize));
			HANDLE_ERROR(cudaMalloc((void **)&dpNodes2, uniNodeSize));
			HANDLE_ERROR(cudaMalloc((void **)&dUniEpn, sizeof(int)*unisum));
			//HANDLE_ERROR(cudaMalloc((void **)&dUniEpn, sizeof(int)*uniEpnSize));
			HANDLE_ERROR(cudaMalloc((void **)&dout5, sizeof(double)*noNodes*scaler * 2));
			HANDLE_ERROR(cudaMalloc((void **)&dNij, sizeof(int)*noNodes*scaler * 54));
			HANDLE_ERROR(cudaMalloc((void **)&dNijk, sizeof(int)*noNodes *scaler * 162));

			//space alloc host
			out5 = (double *)malloc(sizeof(double)*noNodes*scaler * 2);



			//cp to devp'4
			

			HANDLE_ERROR(cudaMemcpy(dpEdges2, pUniEdges, uniEdgeSize, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dpNodes2, pUniNodes, uniNodeSize, cudaMemcpyHostToDevice));
			//HANDLE_ERROR(cudaMemcpy(dUniEpn, pUniEpn, sizeof(int)*uniEpnSize, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dUniEpn, pUniEpn, sizeof(int) * unisum, cudaMemcpyHostToDevice));
			free(pUniNodes); pUniNodes = NULL;
			free(pUniEdges); pUniEdges = NULL;
			free(pUniEpn); pUniEpn = NULL;

			cudaEvent_t run4Start, run4End;
			cudaEventCreate(&run4Start);
			cudaEventCreate(&run4End);
			cudaEventRecord(run4Start, 0);
			float run4Time;

			run4 << <scaler * 2, noNodes >> >(scaler, dUniEpn, genesetlength, edSum, unisum, samples, samples2, dtriA, dtriAb, dpEdges2, dpNodes2, dppn, dstf, dNij, dNijk, dout5, dppnLength);
			//printf("run 4 finished\n");
			cudaEventRecord(run4End, 0);
			cudaEventSynchronize(run4End);
			cudaEventElapsedTime(&run4Time, run4Start, run4End);
			//printf("run 4 time : %f\n", run4Time);
			HANDLE_ERROR(cudaMemcpy(out5, dout5, sizeof(double)*noNodes*scaler * 2, cudaMemcpyDeviceToHost));

			HANDLE_ERROR(cudaFree(dNij)); dNij = NULL;
			HANDLE_ERROR(cudaFree(dNijk)); dNijk = NULL;
			HANDLE_ERROR(cudaFree(dppn)); dppn = NULL;
			HANDLE_ERROR(cudaFree(dstf)); dstf = NULL;
			HANDLE_ERROR(cudaFree(dout5)); dout5 = NULL;
			HANDLE_ERROR(cudaFree(dpEdges2)); dpEdges2 = NULL;
			HANDLE_ERROR(cudaFree(dpNodes2)); dpNodes = NULL;
			HANDLE_ERROR(cudaFree(dUniEpn)); dUniEpn = NULL;
			HANDLE_ERROR(cudaFree(dtriA)); dtriA = NULL;
			HANDLE_ERROR(cudaFree(dtriAb)); dtriAb = NULL;
			
			cudaError_t last = cudaGetLastError();
			if (last != cudaSuccess)
			{
				printf("%s\n", cudaGetErrorString(last));
			}
			//int div = 0;
			// end final cuda run ***********************************************************************

			// begin divergence calc
			double *lval1;
			lval1 = (double *)malloc(sizeof(double)*scaler * 2);

			for (int i = 0; i < scaler * 2; i++)
			{
				lval1[i] = 0.0;

			}

			// compute likelihood of different dataset parsed by 2 iterations 
			for (int g = 0; g < 2; g++){
				int set = 0;
				int place = 0;
				double scoreSum = 0;
				double *likeli1;
				double min = 0;
				double max = 0;
				double inAlpha = 0;
				double probScale = 0;
				double likeliSum = 0;
				double nonInf = 0;
				double *dist;

				double *adjusted;
				double *infFlag;
				double *outq;
				int localoffset;
				outq = out5;
				if (g < 1){

					localoffset = 0;
				}
				else
				{

					localoffset = scaler;

				}

				dist = (double *)malloc(sizeof(double)*scaler);
				likeli1 = (double *)malloc(sizeof(double)*scaler);
				adjusted = (double *)malloc(sizeof(double)*scaler);
				infFlag = (double *)malloc(sizeof(double)*scaler);
				for (int k2 = 0; k2 < scaler; k2++){

					dist[k2] = 0.0;

					adjusted[k2] = 0.0;
					infFlag[k2] = 0.0;

				}
				for (int i = 0; i < scaler*noNodes; i++){

					dist[place] += outq[i + localoffset*noNodes];
					set++;
					if (set == noNodes){
						set = 0;
						place++;
					}
				}

				min = dist[0];
				max = dist[0];
				for (int j3 = 0; j3 < scaler; j3++){
					//			printf(" dis %d %f \n", j3, dist[j3]);
					scoreSum += dist[j3];
					if (dist[j3]>max){

						max = dist[j3];
					}
					if (dist[j3] < min){

						min = dist[j3];
					}

				}


				inAlpha = -1 * (scoreSum / scaler);
				//printf("\n min-%f max-%f", min, max);
				//printf("inAlpha-%f", inAlpha);
				probScale = (10) / (max - min);

				for (int m = 0; m < scaler; m++){

					adjusted[m] = (dist[m] + inAlpha)*probScale;
					//			printf("\n adjusted: %f", adjusted[m]);
					likeli1[m] = exp(adjusted[m]);
					//			printf("\n likeli: %f", likeli1[m]);
					nonInf += likeli1[m];
					//likeLi[m] = posInf;
					//suppress overflow infinity error
					#pragma warning(suppress: 4056)
					if (likeli1[m] >= INFINITY || likeli1[m] <= -INFINITY){
						infFlag[m] = 1.0;
						likeliSum++;
					}

				}
				free(dist); dist = NULL;
				free(adjusted); adjusted = NULL;
				//	printf("\n likesum: %f nonInf: %f", likeliSum, nonInf);

				if (likeliSum == 0){
					for (int meow = 0; meow < scaler; meow++){
						likeli1[meow] = likeli1[meow] / nonInf;
						lval1[meow + localoffset] = likeli1[meow];

					}

				}
				else{
					for (int meow = 0; meow < scaler; meow++){
						likeli1[meow] = infFlag[meow] / likeliSum;
						lval1[meow + localoffset] = likeli1[meow];
					}
				}

				free(likeli1); likeli1 = NULL;
				free(infFlag); infFlag = NULL;
				outq = NULL;
			}

			if (permNum == 0)
			{
				first_lval1 = (double *)malloc(sizeof(double) * scaler * 2);
				memcpy(first_lval1, lval1, sizeof(double) * scaler * 2);
			}


			/*******************************************************************************************************************************************/
			double *sea;
			sea = (double*)malloc(sizeof(double)*scaler);

			//scaler unique number of networks
			for (int i = 0; i < scaler; i++)
			{
				sea[i] = (lval1[i] + lval1[i + scaler]) / 2;

			}

			double js = 0;
			double logger = log(2.0);

			//final score
			js = kool(lval1, sea, 0, scaler) / 2 + kool(lval1, sea, scaler, scaler) / 2;
			//printf("\njs: %f\n", js / logger);
			assert(permNum < perms);
			jsVals[permNum] = js / logger;

			//printf("permutation : %d\n", permNum);
			if (isnan(jsVals[0]))
			{
				printf("jsVal[0] NAN--Breaking permutation loop\n");
				jsVals[0] = -999.0;
				break;
			}
			
			
			

			/****************************time**********************************/
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			cudaEventElapsedTime(&time, start, stop);



			/*************************************************/



			//printf("\n Time1 for the kernel: %f ms\n", time);
			//printf("\n\n");

			free(sea); sea = NULL;
			free(out5); out5 = NULL;
			free(lval1); lval1 = NULL;

			if (permNum % 100 == 0)
			{
				//print every 100 permutations
				printf("Permutation %d finished\n", permNum);
			}

		}//---------------------------------------------------------------------------
		//for loop ends for permutations.
		printf("permutation loop finished\n");
		//free prior knowledge matrix- no longer needed
		
		int nanFlag = 0;
		for (int i = 0; i < perms; i++)
		{
			if (isnan(jsVals[i]))
			{
				nanFlag = 1;
				break;
			}
		}
		if (jsVals[0] < 0 || nanFlag)
		{
			//fprintf(results, "%s %s\n", pathwayName, "GARBAGE VALUE-NAN");
			fprintf(results, "%s failed with %d genes- one of the js scores was nan\n", pathwayName, genesetlength);
			printf("nan value- quiting!\n");
			continue;
		}
		printf("\n\n\n\npermutations finished\n");
		//count how many js values are larger than initial run
		printf("Original JS : %f\n", jsVals[0]);
		/*int largerTally = 0;
		for (int i = 1; i < perms; i++)
		{

			if (jsVals[i] > jsVals[0])
			{
				largerTally++;
			}
		}*/

		//double p_val = 0.0;
		//printf("number larger : %d\n", largerTally);
		//p_val = largerTally / (perms * 1.0);
		//printf("p = %f\n", p_val);
		double p = 0;
		
		if (perms > 0)
		{
			double mu = mean(jsVals, perms);
			double var = variance(mu, jsVals, perms);
			double alpha = (((1 - mu) / var) - (1 / mu)) * pow(mu, 2);
			double beta = alpha * (1 / mu - 1);
			printf("alpha : %f beta : %f\n", alpha, beta);
			//p = 1 - betaCDF(jsVals[0], alpha, beta);
			p = boost::math::ibetac(alpha, beta, jsVals[0]);

			printf("p = %f\n", p);
			fprintf(results, "%s %f %f %d\n", pathwayName, jsVals[0], p, genesetlength);
		}
		
		if (p < pThreshold) //statistically significant
		{
			
			char networkFilePath[600];
			strcpy(networkFilePath, pathwayName);
			strcat(networkFilePath, "_Networks.txt");
			char bdeuFilePath[600];
			strcpy(bdeuFilePath, pathwayName);
			strcat(bdeuFilePath, "_BDEU_SCORES.txt");

			writeNetworkFile(networkFilePath, inputFile, classFile, pathwayName, first_unisum, first_uniEpn, genesetgenes, genesetlength, first_uniNodes, first_uniEdges, first_numEdges, uniqueNetIds);
			writeBdeuScores(bdeuFilePath, inputFile, classFile, pathwayName, class1, class2, first_scaler, first_lval1);

			//printf("\nOriginal JS score : %f\n", jsVals[0]);
			//printf("Original number of unique networks : %d\n", first_unisum);

			//-----------------------------------------------------
			//edgeList calcs
			//edgesPerNetworkKernel --> run22 --> output Edge list
			//NOTE: genesetlength is used to represent the number of nodes aka noNodes as used in prior kernel calls
			//printf("Final run\n");

			//2 networks are being looked at and c = number of different gene combinations
			int c = (((genesetlength*genesetlength) - genesetlength) / 2);
			int numNetworks = 2;

			//host copies
			int *nodes, *edges, *out23;
			int edgesPN[3];

			out23 = (int *)malloc(sizeof(int) * c * numNetworks);

			//dev copies
			int *dout23, *dsrchAry, *dEdgesPN;

			//copy data taken from first run2 permutation
			int *ptr1 = &out23[0];
			int *ptr2 = &out23[c];
			memcpy(ptr1, edgeListData1, sizeof(int) * c);
			memcpy(ptr2, edgeListData2, sizeof(int) * c);
			ptr1 = NULL;
			ptr2 = NULL;

			//allocate device memory and copy
			HANDLE_ERROR(cudaMalloc((void **)&dout23, sizeof(int) * c * numNetworks));
			HANDLE_ERROR(cudaMalloc((void **)&dEdgesPN, sizeof(int) * (numNetworks + 1)));
			HANDLE_ERROR(cudaMalloc((void **)&dsrchAry, genesetlength * sizeof(int)));

			HANDLE_ERROR(cudaMemcpy(dout23, out23, sizeof(int) * c * numNetworks, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dsrchAry, initialSearcher, genesetlength * sizeof(int), cudaMemcpyHostToDevice));

			const int PARENTS_LIMIT = INT_MAX;
			
			//edgePerNetworkKernel << <numNetworks + 1, c, (c * sizeof(int)) >> >(dout23, dEdgesPN, dsrchAry, genesetlength, PARENTS_LIMIT, c);
			edgePerNetworkKernel << <numNetworks + 1, 1 >> >(dout23, dEdgesPN, dsrchAry, genesetlength, PARENTS_LIMIT, c);
			//edgePerNetworkKernel << <numNetworks + 1, c, (c * sizeof(int)) + (genesetlength * sizeof(int)) >> >(dout23, dEdgesPN, dsrchAry, genesetlength, PARENTS_LIMIT, c);
			
			//edgePerNetworkKernel calculates sum of edges for each network - now we need to perform the prefix calc for edgesPN on the CPU
			int tempEdgeSum[numNetworks + 1];
			HANDLE_ERROR(cudaMemcpy(tempEdgeSum, dEdgesPN, sizeof(int) * (numNetworks + 1), cudaMemcpyDeviceToHost));
			edgesPN[0] = 0;
			//calc prefix sum
			for(int i = 1; i < numNetworks + 1; i++){
				edgesPN[i] = edgesPN[i-1] + tempEdgeSum[i-1];
			}//copy results of prefix sum back to GPU for use in run22
			HANDLE_ERROR(cudaMemcpy(dEdgesPN, edgesPN, sizeof(int) * (numNetworks + 1), cudaMemcpyHostToDevice));


			/*for (int i = 0; i < 3; i++)
			{
				printf("edgesPN[%d]  : %d\n", i, edgesPN[i]);
			}
			
			for (int i = 0; i < genesetlength; i++)
			{
				printf("%d : %s\n", i, genesetgenes[i]);
			}*/



			//needed to calculate how long to make edge array
			int totalEdges = edgesPN[2];

			nodes = (int *)malloc(sizeof(int) * genesetlength * 2);
			edges = (int *)malloc(sizeof(int) * totalEdges);


			int *dNodes, *dEdges;
			HANDLE_ERROR(cudaMalloc((void **)&dNodes, sizeof(int) * genesetlength * 2));
			HANDLE_ERROR(cudaMalloc((void **)&dEdges, sizeof(int) * totalEdges));

			run22 << <numNetworks, genesetlength >> >(c, dEdgesPN, dout23, dNodes, genesetlength, totalEdges, dsrchAry, dEdges, PARENTS_LIMIT);

			HANDLE_ERROR(cudaMemcpy(nodes, dNodes, sizeof(int) * 2 * genesetlength, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(edges, dEdges, sizeof(int) * totalEdges, cudaMemcpyDeviceToHost));


			HANDLE_ERROR(cudaFree(dout23)); dout23 = NULL;
			HANDLE_ERROR(cudaFree(dsrchAry)); dsrchAry = NULL;
			HANDLE_ERROR(cudaFree(dEdgesPN)); dEdgesPN = NULL;
			HANDLE_ERROR(cudaFree(dNodes)); dNodes = NULL;
			HANDLE_ERROR(cudaFree(dEdges)); dEdges = NULL;
			free(out23); out23 = NULL;

			char edgeListFile[600];
			strcpy(edgeListFile, pathwayName);
			strcat(edgeListFile, "_EdgeList.txt");
			//int networkIds[2] = { 1, 2 };
			//writeNetworkFile(edgeListFile, inputFile, classFile, pathwayName, 2, edgesPN, genesetgenes, genesetlength, nodes, edges, totalEdges, networkIds);
			writeEdgeListFile(edgeListFile, inputFile, classFile, pathwayName, genesetgenes, genesetlength, nodes, edges, edgesPN, priorMatrix, class1, class2);
			
			cudaEventRecord(end, 0);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&totalTime, begin, end);

			//printf("Total Run Time : %f\n", totalTime);

			//FILE *timeFile = fopen("Time.txt", "w");
			//fprintf(timeFile, "%f", totalTime);
			//output number of unique networks to check against java scores
			printf("Total run time : %f\n", totalTime);
			//fprintf(timeFile, "%f", totalTime);
			//fclose(timeFile);
			free(nodes); nodes = NULL;
			free(edges); edges = NULL;
		}

		printf("\nPathway finished.\n\n");
		free(priorMatrix); priorMatrix = NULL; //free this after writing files b/c needed for writeEdgeListFile

		diff = clock() - cpuTime;
		int msec = diff * 1000 / CLOCKS_PER_SEC;
		printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
		//---------------------------------------------------------------------------
		//free variables

		

		free(transferdata1);
		free(transferdata2);
		transferdata1 = NULL;
		transferdata2 = NULL;

		free(first_lval1);
		free(first_uniNodes);
		free(first_uniEdges);
		free(first_uniEpn);
		free(jsVals);
		free(uniqueNetIds);
		free(edgeListData1);
		free(edgeListData2);
		//free(initialFF);
		free(initialSearcher);
		//free(initialSpacr);
		first_lval1 = NULL;
		first_uniNodes = NULL;
		first_uniEdges = NULL;
		first_uniEpn = NULL;
		jsVals = NULL;
		uniqueNetIds = NULL;
		edgeListData1 = NULL;
		edgeListData2 = NULL;
		//initialFF = NULL;
		initialSearcher = NULL;
		//initialSpacr = NULL;

	}
	fclose(fp3);
	fclose(results);

	free(data);
	data = NULL;
	//use to make sure all data is recorded and visual
	//profiler works
	//cudaDeviceReset();

	return 0;
}
