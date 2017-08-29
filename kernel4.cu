#include "definitions.cuh"
#include "assert.h"
#include <stdio.h>

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

