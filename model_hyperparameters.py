class ModelHyperparameters:
    learning_rate = 0.001
    mini_batch_size = 32
    neurons_per_hidden_layer = [500, 50]
    dropout_rate = 0.2

    # One of 'RMSProp' and 'Adam'
    optimizer = 'RMSProp'
    # Only applies when using 'RMSProp' optimizer
    rms_prop_rho = 0.9

    use_batch_normalization = True
    use_l2_regularization = False