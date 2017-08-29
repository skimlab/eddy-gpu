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

