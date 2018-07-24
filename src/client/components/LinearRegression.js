import React, { Component } from 'react';

import { generateData } from '../data/LinearRegression';
import { TrainLinearRegression } from '../train/LinearRegression';
import ScatterChartWithData from './ScatterChartWithData';

class LinearRegression extends Component {
  constructor() {
    super();
    this.state = {
      values: [],
      randomCoefficients: [],
      predictedCoefficients: [],
      beforePrediction: [],
      afterPrediction: [],
    };
    const trueCoefficients = { a: .2, b: 0.5 };
    this.trainingData = generateData(100, trueCoefficients);
    this.numIterations = 75;
    this.learningRate = 0.5;
    this.trainLinearRegression = new TrainLinearRegression(this.learningRate);
  }

  async getData() {
    const xvals = await this.trainingData.xs.data();
    const yvals = await this.trainingData.ys.data();

    const values = Array.from(yvals).map((y, i) => {
      return { 'x': xvals[i], 'y': yvals[i] };
    });
    return values;
  }

  getValues(values) {
    this.setState({
      values
    });
  }

  async getPredictionAndData(after) {
    const xvals = await this.trainingData.xs.data();
    const predVals = await this.trainLinearRegression.learnCoefficients(this.trainingData, this.numIterations, after);
    const values = Array.from(xvals).map((y, i) => {
      return { 'x': xvals[i], y: predVals[i] };
    });
    return values;
  }

  getBeforePredictedValue(values) {
    const a = this.trainLinearRegression.a.dataSync()[0].toFixed(2);
    const b = this.trainLinearRegression.b.dataSync()[0].toFixed(2);
    this.setState({
      beforePrediction: values,
      randomCoefficients: { a, b },
    });
  }

  getAfterPredictedValue(values) {
    const a = this.trainLinearRegression.a.dataSync()[0].toFixed(2);
    const b = this.trainLinearRegression.b.dataSync()[0].toFixed(2);
    this.setState({
      afterPrediction: values,
      predictedCoefficients: { a, b },
    });
  }

  componentDidMount() {
    this.getData().then((values) => {
      this.getValues(values);
    }).catch((err) => {
      console.error(err);
    });

    this.getPredictionAndData().then((values) => {
      this.getBeforePredictedValue(values);
    }).catch((err) => {
      console.error(err);
    });

    this.getPredictionAndData(true).then((values) => {
      this.getAfterPredictedValue(values);
    }).catch((err) => {
      console.error(err);
    });
  }

  render() {
    return (
      <div>
        <h1>LinearRegression</h1>
        <div>
          <ScatterChartWithData values={this.state.values} />
        </div>
        <div>
          <p>Random Coefficients</p>
          <p>a:{this.state.randomCoefficients.a}</p>
          <p>b:{this.state.randomCoefficients.b}</p>
          <ScatterChartWithData values={this.state.values} prediction={this.state.beforePrediction} />
        </div>
        <div>
          <p>Predicted Coefficients</p>
          <p>a:{this.state.predictedCoefficients.a}</p>
          <p>b:{this.state.predictedCoefficients.b}</p>
          <ScatterChartWithData values={this.state.values} prediction={this.state.afterPrediction} />
        </div>
      </div>
    );
  }
}

export default LinearRegression;