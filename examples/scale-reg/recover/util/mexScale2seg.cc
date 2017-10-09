#include "mex.h"
#include "matrix.h"
#include "omp.h"
bool out_range(int x, int y, int w, int h) {
	if (x < 0 || x >= w || y < 0 || y >= h)
		return true;
	else
		return false;
}

float l2(int x1, int y1, int x2, int y2) {
	return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs != 1)mexErrMsgTxt("1 input arg required");
	if (nrhs != 1)mexErrMsgTxt("1 output arg required");
	if (mxGetNumberOfDimensions(prhs[0]) != 2)mexErrMsgTxt("Input arg require 2D matrix");
	if (!mxIsSingle(prhs[0]))mexErrMsgTxt("Input arg require type single");
	int h = mxGetM(prhs[0]); int w = mxGetN(prhs[0]);
	//mexPrintf("%d, %d\n", w, h);
	float *data = (float*)mxGetData(prhs[0]); 
	plhs[0] = mxCreateNumericMatrix(h, w, mxSINGLE_CLASS, mxREAL);
	float *o = (float*)mxGetData(plhs[0]);
	//mexPrintf("Input matrix size: %d, %d\n", h, w);
    #pragma omp parallel for
    for (int x = 0; x < w; ++x) {
    	for (int y = 0; y < h; ++y) {
    		float s = data[x * h + y];
    		int r = int (s / 2);
    		if (r != 0) { 
    			for (int x1 = x - r; x1 <= x+r; ++x1) {
					for (int y1 = y - r; y1 <= y + r; ++y1) {
						if (!out_range(x1, y1, w, h)) {
							if (l2(x1, y1, x, y) <= r*r){
 								o[x1 * h + y1] = float(1.0);
							}
						}
					}    				
    			}
    		}
    	}
    }
    return ;
}