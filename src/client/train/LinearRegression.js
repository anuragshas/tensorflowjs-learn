import * as tf from '@tensorflow/tfjs';

export class TrainLinearRegression {
  constructor(learningRate = 0.5) {
    /**
 * We want to learn the coefficients that give correct solutions to the
 * following cubic equation:
 *      y = a * x + b
 * In other words we want to learn values for:
 *      a
 *      b
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */

    // Step 1. Set up variables, these are the things we want the model
    // to learn in order to do prediction accurately. We will initialize
    // them with random values.
    this.a = tf.variable(tf.scalar(Math.random()));
    this.b = tf.variable(tf.scalar(Math.random()));
    // Step 2. Create an optimizer, we will use this later. You can play
    // with some of these values to see how the model performs.
    this.learningRate = learningRate;
    this.optimizer = tf.train.sgd(this.learningRate);
  }
  // Step 3. Write our training process functions.

  /*
   * This function represents our 'model'. Given an input 'x' it will try and
   * predict the appropriate output 'y'.
   *
   * It is also sometimes referred to as the 'forward' step of our training
   * process. Though we will use the same function for predictions later.
   *
   * @return number predicted y value
   */
  predict(x) {
    // y = a * x  + b
    return tf.tidy((() => {
      return this.a.mul(x)
        .add(this.b);
    }).bind(this));
  }

  /*
   * This will tell us how good the 'prediction' is given what we actually
   * expected.
   *
   * prediction is a tensor with our predicted y values.
   * labels is a tensor with the y values the model should have predicted.
   */
  loss(prediction, labels) {
    // Having a good error function is key for training a machine learning model
    const error = prediction.sub(labels).square().mean();
    return error;
  }

  /*
   * This will iteratively train our model.
   *
   * xs - training data x values
   * ys — training data y values
   */
  async train(xs, ys, numIterations) {
    for (let iter = 0; iter < numIterations; iter++) {
      // optimizer.minimize is where the training happens.

      // The function it takes must return a numerical estimate (i.e. loss)
      // of how well we are doing using the current state of
      // the variables we created at the start.

      // This optimizer does the 'backward' step of our training process
      // updating variables defined previously in order to minimize the
      // loss.
      this.optimizer.minimize((() => {
        // Feed the examples into the model
        const pred = this.predict(xs);
        return this.loss(pred, ys);
      }).bind(this));

      // Use tf.nextFrame to not block the browser.
      await tf.nextFrame();
    }
  }

  async learnCoefficients(trainingData, numIterations, after = false) {
    if (after) {
      // Train the model!
      await this.train(trainingData.xs, trainingData.ys, numIterations);
    }

    const predictions = this.predict(trainingData.xs);
    const predictionData = await predictions.data();
    predictions.dispose();
    return predictionData;
  }
}
