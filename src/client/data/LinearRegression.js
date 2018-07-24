import * as tf from '@tensorflow/tfjs';

export function generateData(numPoints, coeff, sigma = 0.1) {
  return tf.tidy(() => {
    const [a, b] = [
      tf.scalar(coeff.a), tf.scalar(coeff.b)
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    // Generate data
    const ys = a.mul(xs)
      .add(b)
      // Add random noise to the generated data
      // to make the problem a bit more interesting
      .add(tf.randomNormal([numPoints], 0, sigma));

    // Normalize the y values to the range 0 to 1.
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys
    };
  });
}