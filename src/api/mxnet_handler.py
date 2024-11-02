from flask import request, jsonify

import mxnet as mx
import raven_ai.mxnet_ai.mxnet_trainer
import raven_ai.mxnet_ai.mxnet_model
import raven_ai.data_handler

def train():
    try:
        # Parse input JSON data
        data = request.get_json()
        data_path = data.get("data_path", "training_data.npz")  # Use provided data path or default

        # Initialize the data handler
        data_handler = raven_ai.data_handler.DataHandler(data_path)

        # Initialize the LSTM model
        model = raven_ai.mxnet_ai.mxnet_model.SimpleLSTM(hidden_size=64, num_layers=2)
        model.initialize(mx.init.Xavier())

        # Create and run the LSTMTrainer
        trainer = raven_ai.mxnet_ai.mxnet_trainer.LSTMTrainer(model=model, data_handler=data_handler, config={"batch_size": 5, "epochs": 5, "learning_rate": 5})
        trainer.train()  # This will print the training process

        return jsonify({"status": "training complete"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500