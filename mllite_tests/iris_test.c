#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <xgboost/c_api.h>

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
  const char *name = "data/original/iris.csv?format=csv";
  DMatrixHandle train_handle;
  safe_xgboost(XGDMatrixCreateFromFile(name, 1, &train_handle));
  DMatrixHandle dmats[1] = {train_handle};
  // CREATE_ BOOSTER
  BoosterHandle booster;
  safe_xgboost(XGBoosterCreate(dmats, 1, &booster));
  safe_xgboost(XGBoosterSetParam(booster, "nthread", "1"));
  safe_xgboost(XGBoosterSetParam(booster, "device", "cpu"));
  safe_xgboost(XGBoosterSetParam(booster, "booster", "gbtree"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
  safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1"));
  // TRAIN
  
  const char* eval_names[1] = {"train"};
  const char* eval_result = NULL;
  int num_of_iterations = 20;
  for (int i = 0; i < num_of_iterations; ++i) {
    // Update the model performance for each iteration    
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, train_handle));
    safe_xgboost(XGBoosterEvalOneIter(booster, i, dmats, eval_names, 2, &eval_result));
    printf("%s\n", eval_result);
  }
  
  bst_ulong num_of_features = 0;
  XGBoosterGetNumFeature(booster, &num_of_features);
  printf("NUM_FEATURES = %ld", num_of_features);
 
  return 0;
}
