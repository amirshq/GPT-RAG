def resnet_block(features, bottleneck, out_filters, training):
    """Residual block."""
    with tf.variable_scope("input"):
        original = features
        features = tf.layers.conv2d(features, bottleneck, 1, activation=None)
        features = tf.layers.batch_normalization(features, training=training)
        features = tf.nn.relu(features)

    with tf.variable_scope("bottleneck"):
        features = tf.layers.conv2d(
            features, bottleneck, 3, activation=None, padding="same")
        features = tf.layers.batch_normalization(features, training=training)
        features = tf.nn.relu(features)

    with tf.variable_scope("output"):
        features = tf.layers.conv2d(features, out_filters, 1)
        in_dims = original.shape[-1].value
        if in_dims != out_filters:
            original = tf.layers.conv2d(features, out_filters, 1, activation=None,
                name="proj")
        features += original
    return features