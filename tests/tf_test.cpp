#include <gtest/gtest.h>
#include <tensorflow/c/c_api.h>
#include "../tf_nn/fen_to_input.hpp"
#include "../tf_nn/get_model_prediction.hpp"

#include <string>

TEST(Tensorflow, Version) { 
    std::string version = TF_Version();
    EXPECT_EQ(version, "2.16.2");
}

TEST(Tensorflow, GetPrediction) { 
    std::string fen = "r2qk3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1";
    float static_eval = 2.390f;
    auto input = fenToInput(fen, static_eval);
    float prediction = predict(input, "test");
    //Way to big (I hope) to be seen in an actual evaluation
    float dummyEvaluation = 10000.0f;
    EXPECT_NE(prediction, dummyEvaluation);
}