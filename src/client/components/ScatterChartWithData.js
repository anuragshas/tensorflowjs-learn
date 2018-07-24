import React from 'react';
import { ScatterChart, CartesianGrid, XAxis, YAxis, Tooltip, Scatter } from 'recharts';

const renderNoShape = (props) => {
  return null;
}

const ScatterChartWithData = (props) => (
  <ScatterChart width={300} height={300}
    margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey={'x'}
      name="x"
      type="number" />
    <YAxis dataKey={'y'}
      name="y"
      domain={[0, 1]}
      type="number" />
    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
    <Scatter name="Points" data={props.values} fill="#82ca9d" />
    {props.prediction &&
      <Scatter name="Line"
        data={props.prediction}
        legendType="line" shape={renderNoShape}
        line={{ stroke: 'red', strokeWidth: 1 }} />}
  </ScatterChart>
);

export default ScatterChartWithData;
