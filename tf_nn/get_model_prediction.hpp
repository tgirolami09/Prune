#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>

// Required to avoid stripping TF symbols
extern "C" void dummy() { TF_Version(); }

#ifndef TF_INPUT_SIZE_VAR
#define TF_INPUT_SIZE_VAR
// 768 board + 1 turn + 1 static_eval
const int TF_INPUT_SIZE = 770; 
#endif

float predict(std::vector<float> input, const std::string modelLocation) {
    std::cout << "TensorFlow C API version: " << TF_Version() << std::endl;

    // ===== Create session options =====
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();

    // ===== Load the SavedModel =====
    TF_Graph* graph = TF_NewGraph();
    const char* tags[] = {"serve"}; // convention for SavedModels
    TF_Session* session = TF_LoadSessionFromSavedModel(
        sess_opts, nullptr, "test", tags, 1, graph, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error loading model: " << TF_Message(status) << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully." << std::endl;

    // ===== Prepare input tensor =====
    std::vector<float> input_data(TF_INPUT_SIZE, 0.0f);
    if (input.size() == TF_INPUT_SIZE){
        input_data = input;
    }
    else{
        printf("Wrong input data size\n");
        // Just for fun pi 
        return 3.1415926536f;
    }

    int64_t dims[2] = {1, TF_INPUT_SIZE};
    TF_Tensor* input_tensor = TF_NewTensor(
        TF_FLOAT, dims, 2,
        input_data.data(), input_data.size() * sizeof(float),
        [](void*, size_t, void*) {}, nullptr);

    // ===== Get input/output operations =====
    TF_Operation* input_op = TF_GraphOperationByName(graph, "serve_board");
    TF_Operation* output_op = TF_GraphOperationByName(graph, "StatefulPartitionedCall");

    if (!input_op || !output_op) {
        std::cerr << "Error: Could not find input or output operations in graph." << std::endl;
        return 1;
    }

    TF_Output inputs[] = { {input_op, 0} };
    TF_Tensor* input_values[] = { input_tensor };
    TF_Output outputs[] = { {output_op, 0} };
    TF_Tensor* output_values[1] = { nullptr };

    // ===== Run session =====
    TF_SessionRun(
        session,
        nullptr, // run options
        inputs, input_values, 1, // inputs
        outputs, output_values, 1, // outputs
        nullptr, 0, // no targets
        nullptr, // run metadata
        status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error running session: " << TF_Message(status) << std::endl;
        return 1;
    }

    // ===== Extract prediction =====
    float* out_data = static_cast<float*>(TF_TensorData(output_values[0]));
    std::cout << "Model prediction: " << out_data[0] << std::endl;

    // ===== Cleanup =====
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_values[0]);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);

    return out_data[0];
}
