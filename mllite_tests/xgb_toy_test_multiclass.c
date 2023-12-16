#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <xgboost/c_api.h>

const int ROWS=1024, COLS=12;


// https://xgboost.readthedocs.io/en/stable/dev/c-api-demo_8c-example.html#a1
// https://xgboost.readthedocs.io/en/stable/tutorials/c_api_tutorial.html

#define safe_xgboost(call) {  \
  int err = (call); \
  if (err != 0) { \
    fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError());  \
    exit(1); \
  } \
}

int main() {

  float data[ROWS*COLS];
  for(size_t i=0; i< ROWS; ++i ) {
    for(size_t j=0; j< COLS; ++j ) {
      data[i + j*ROWS] = i + j;
    }
  }

  DMatrixHandle dmatrix;
  safe_xgboost(XGDMatrixCreateFromMat(data, ROWS, COLS, -1, &dmatrix));
  // variable to store labels for the dataset created from above matrix
  float labels[ROWS];
  for (int i = 0; i < ROWS; i++) {
    labels[i] =  i % 4;
  }
  // Loading the labels
  safe_xgboost(XGDMatrixSetFloatInfo(dmatrix, "label", labels, ROWS));

  DMatrixHandle train_handle = dmatrix;
  DMatrixHandle dmats[1] = {train_handle};
  // CREATE_ BOOSTER
  BoosterHandle booster;
  safe_xgboost(XGBoosterCreate(dmats, 1, &booster));
  safe_xgboost(XGBoosterSetParam(booster, "nthread", "1"));
  safe_xgboost(XGBoosterSetParam(booster, "device", "cpu"));
  safe_xgboost(XGBoosterSetParam(booster, "booster", "gbtree"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
  safe_xgboost(XGBoosterSetParam(booster, "max_bin", "16"));
  safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1"));
  safe_xgboost(XGBoosterSetParam(booster, "num_class", "4"));
  safe_xgboost(XGBoosterSetParam(booster, "objective", "multi:softmax"));
  // TRAIN
  
  const char* eval_names[1] = {"train"};
  const char* eval_result = NULL;
  int num_of_iterations = 200;
  for (int i = 0; i < num_of_iterations; ++i) {
    // Update the model performance for each iteration    
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, train_handle));
    safe_xgboost(XGBoosterEvalOneIter(booster, i, dmats, eval_names, 1, &eval_result));
    printf("EVAL_RESULT_AT_ITERATION %d %s\n", i + 1, eval_result);
  }
  
  bst_ulong num_of_features = 0;
  XGBoosterGetNumFeature(booster, &num_of_features);
  printf("NUM_FEATURES = %ld\n\n", num_of_features);
 
  safe_xgboost(XGBoosterFree(booster));
  return 0;
}
