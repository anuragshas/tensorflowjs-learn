import React, { Component } from 'react';

import { generateData } from '../data/LinearRegression';
import { learnCoefficients } from '../train/LinearRegression';
import ScatterChartWithData from './ScatterChartWithData';

class LinearRegression extends Component {
  constructor() {
    super();
    this.state = {
      values: [],
      beforePrediction: [],
      afterPrediction: [],
    };
    const trueCoefficients = { a: .2, b: -0.5 };
    this.trainingData = generateData(100, trueCoefficients);
    this.numIterations = 75;
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
    const predVals = await learnCoefficients(this.trainingData, this.numIterations, after);
    const values = Array.from(xvals).map((y, i) => {
      return { 'x': xvals[i], y: predVals[i] };
    });
    return values;
  }

  getBeforePredictedValue(values) {
    this.setState({
      beforePrediction: values
    });
  }

  getAfterPredictedValue(values) {
    this.setState({
      afterPrediction: values
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
        <ScatterChartWithData values={this.state.values} />
        <ScatterChartWithData values={this.state.values} prediction={this.state.beforePrediction} />
        <ScatterChartWithData values={this.state.values} prediction={this.state.afterPrediction} />
      </div>
    );
  }
}

export default LinearRegression;