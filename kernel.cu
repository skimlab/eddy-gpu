/*
*	kernel.cu - stores all of the GPU methods used by main.cu
*	TB
*
*/


#include "definitions.cuh"
#include "assert.h"
#include <stdio.h>
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


__global__  void run25(int scaler1, int scaler2, int noEdges, int gLength, int scalerCombo, int *dedgesPN, int  *dNodes, int *dedgeAry, int *shrunk, int *scalerTest, int *shnkplc){
	int index = threadIdx.x + blockDim.x*blockIdx.x;

	//do we need this
	if (index < scalerCombo){

		shrunk[index] = pass1(index, noEdges, scaler1, scaler2, gLength, dNodes, dedgeAry, dedgesPN, scalerTest[index], shnkplc);

	}

}

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

