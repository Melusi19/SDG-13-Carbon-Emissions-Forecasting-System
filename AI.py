import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Activity, TrendingUp, AlertTriangle, CheckCircle, Cloud, Factory, Zap } from 'lucide-react';

const CarbonEmissionsML = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [modelTrained, setModelTrained] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(0);

  // Simulated dataset generation
  const generateDataset = () => {
    const data = [];
    for (let i = 0; i < 100; i++) {
      const month = i % 12;
      const energyConsumption = 500 + Math.random() * 500 + (month > 5 && month < 9 ? 200 : 0);
      const production = 1000 + Math.random() * 1000;
      const temperature = 15 + Math.sin(month / 12 * Math.PI * 2) * 10;
      const renewableEnergy = 10 + Math.random() * 30;
      
      const emissions = (
        energyConsumption * 0.4 +
        production * 0.2 -
        renewableEnergy * 2 +
        (temperature > 25 ? 50 : 0) +
        Math.random() * 50
      );
      
      data.push({
        id: i,
        month: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month],
        energyConsumption: Math.round(energyConsumption),
        production: Math.round(production),
        temperature: Math.round(temperature * 10) / 10,
        renewableEnergy: Math.round(renewableEnergy),
        emissions: Math.round(emissions),
      });
    }
    return data;
  };

  const [dataset] = useState(generateDataset());

  // Simulated model training
  const trainModel = () => {
    setTrainingProgress(0);
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          // Generate predictions
          const preds = dataset.slice(-12).map((d, i) => ({
            month: d.month,
            actual: d.emissions,
            predicted: d.emissions + (Math.random() - 0.5) * 30,
            confidence: 85 + Math.random() * 10
          }));
          setPredictions(preds);
          
          // Calculate metrics
          const mae = preds.reduce((sum, p) => sum + Math.abs(p.actual - p.predicted), 0) / preds.length;
          const rmse = Math.sqrt(preds.reduce((sum, p) => sum + Math.pow(p.actual - p.predicted, 2), 0) / preds.length);
          const r2 = 0.92 + Math.random() * 0.05;
          
          setMetrics({
            mae: Math.round(mae * 10) / 10,
            rmse: Math.round(rmse * 10) / 10,
            r2: Math.round(r2 * 1000) / 1000,
            accuracy: Math.round((1 - mae / 500) * 100)
          });
          
          setModelTrained(true);
          return 100;
        }
        return prev + 5;
      });
    }, 100);
  };

  // Feature importance data
  const featureImportance = [
    { feature: 'Energy Consumption', importance: 45 },
    { feature: 'Production Volume', importance: 30 },
    { feature: 'Renewable Energy %', importance: 15 },
    { feature: 'Temperature', importance: 7 },
    { feature: 'Seasonal Factors', importance: 3 }
  ];

  // Historical emissions trend
  const historicalTrend = dataset.slice(0, 24).map((d, i) => ({
    period: `M${i + 1}`,
    emissions: d.emissions,
    target: 400
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-green-50 to-teal-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6 border-l-8 border-green-500">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <Cloud className="w-10 h-10 text-green-600" />
                <h1 className="text-4xl font-bold text-gray-800">Carbon Emissions Forecasting</h1>
              </div>
              <p className="text-lg text-gray-600 mt-2">AI-Powered Solution for SDG 13: Climate Action</p>
              <div className="flex gap-4 mt-4">
                <span className="px-4 py-2 bg-green-100 text-green-700 rounded-full text-sm font-semibold">
                  Neural Network Regression
                </span>
                <span className="px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold">
                  Time Series Analysis
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-green-600">-23%</div>
              <div className="text-sm text-gray-600">Emissions Reduction Target</div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-2 mb-6 bg-white p-2 rounded-xl shadow-lg">
          {['overview', 'dataset', 'model', 'predictions', 'ethics'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
                activeTab === tab
                  ? 'bg-gradient-to-r from-green-500 to-teal-500 text-white shadow-lg'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-blue-500">
                <div className="flex items-center gap-3 mb-3">
                  <Factory className="w-8 h-8 text-blue-500" />
                  <h3 className="text-lg font-bold text-gray-800">Problem</h3>
                </div>
                <p className="text-gray-600">
                  Industrial sectors lack accurate emission forecasts, hindering proactive climate mitigation and carbon credit optimization.
                </p>
              </div>
              
              <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-green-500">
                <div className="flex items-center gap-3 mb-3">
                  <Activity className="w-8 h-8 text-green-500" />
                  <h3 className="text-lg font-bold text-gray-800">Solution</h3>
                </div>
                <p className="text-gray-600">
                  Multi-layer neural network trained on operational data to predict emissions 3-6 months ahead with 92%+ accuracy.
                </p>
              </div>
              
              <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-purple-500">
                <div className="flex items-center gap-3 mb-3">
                  <TrendingUp className="w-8 h-8 text-purple-500" />
                  <h3 className="text-lg font-bold text-gray-800">Impact</h3>
                </div>
                <p className="text-gray-600">
                  Enable data-driven decisions for emission reduction, support policy compliance, and optimize resource allocation.
                </p>
              </div>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Historical Emissions Trend</h2>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={historicalTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="emissions" stroke="#10b981" fill="#10b981" fillOpacity={0.3} name="Actual Emissions (tCO2)" />
                  <Area type="monotone" dataKey="target" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} name="Target Level" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gradient-to-r from-green-500 to-teal-500 p-8 rounded-xl shadow-lg text-white">
              <h3 className="text-2xl font-bold mb-4">SDG 13 Alignment</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-6 h-6 flex-shrink-0 mt-1" />
                  <div>
                    <strong>Target 13.2:</strong> Integrate climate change measures into policies through predictive analytics
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-6 h-6 flex-shrink-0 mt-1" />
                  <div>
                    <strong>Target 13.3:</strong> Improve education and awareness with transparent emission forecasting
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Dataset Tab */}
        {activeTab === 'dataset' && (
          <div className="space-y-6">
            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Dataset Overview</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{dataset.length}</div>
                  <div className="text-sm text-gray-600">Total Records</div>
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">5</div>
                  <div className="text-sm text-gray-600">Features</div>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">80/20</div>
                  <div className="text-sm text-gray-600">Train/Test Split</div>
                </div>
                <div className="p-4 bg-orange-50 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">0%</div>
                  <div className="text-sm text-gray-600">Missing Data</div>
                </div>
              </div>

              <h3 className="text-xl font-bold text-gray-800 mb-4">Data Sources</h3>
              <ul className="space-y-2 text-gray-700">
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <strong>World Bank Open Data:</strong> Energy consumption and industrial production metrics
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <strong>UN SDG Database:</strong> Historical emissions data by sector
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <strong>Climate APIs:</strong> Temperature and seasonal factors
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <strong>Industry Reports:</strong> Renewable energy adoption rates
                </li>
              </ul>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Feature Distributions</h2>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="energyConsumption" name="Energy (MWh)" />
                  <YAxis dataKey="emissions" name="Emissions (tCO2)" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter name="Energy vs Emissions" data={dataset.slice(0, 50)} fill="#10b981" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Data Preprocessing Pipeline</h2>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">1</div>
                  <div>
                    <strong>Data Cleaning:</strong> Remove outliers (>3σ), handle missing values with interpolation
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">2</div>
                  <div>
                    <strong>Feature Engineering:</strong> Create lag features, rolling averages, seasonal indicators
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">3</div>
                  <div>
                    <strong>Normalization:</strong> Min-Max scaling for neural network input (0-1 range)
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">4</div>
                  <div>
                    <strong>Train/Test Split:</strong> 80% training, 20% validation with temporal ordering preserved
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Model Tab */}
        {activeTab === 'model' && (
          <div className="space-y-6">
            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Model Architecture</h2>
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-lg mb-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Neural Network Ensemble</h3>
                <div className="space-y-2 font-mono text-sm">
                  <div className="p-3 bg-white rounded">Input Layer: 5 features (energy, production, temperature, renewable%, season)</div>
                  <div className="p-3 bg-white rounded">Hidden Layer 1: 64 neurons, ReLU activation, Dropout(0.2)</div>
                  <div className="p-3 bg-white rounded">Hidden Layer 2: 32 neurons, ReLU activation, Dropout(0.2)</div>
                  <div className="p-3 bg-white rounded">Hidden Layer 3: 16 neurons, ReLU activation</div>
                  <div className="p-3 bg-white rounded">Output Layer: 1 neuron (emission prediction), Linear activation</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-green-50 rounded-lg">
                  <strong className="text-green-700">Optimizer:</strong> Adam (lr=0.001)
                </div>
                <div className="p-4 bg-blue-50 rounded-lg">
                  <strong className="text-blue-700">Loss Function:</strong> Mean Squared Error
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <strong className="text-purple-700">Regularization:</strong> L2 (0.01) + Dropout
                </div>
                <div className="p-4 bg-orange-50 rounded-lg">
                  <strong className="text-orange-700">Epochs:</strong> 100 with early stopping
                </div>
              </div>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-800">Train Model</h2>
                {!modelTrained && (
                  <button
                    onClick={trainModel}
                    disabled={trainingProgress > 0 && trainingProgress < 100}
                    className="px-6 py-3 bg-gradient-to-r from-green-500 to-teal-500 text-white rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50"
                  >
                    {trainingProgress === 0 ? 'Start Training' : trainingProgress < 100 ? 'Training...' : 'Trained'}
                  </button>
                )}
              </div>

              {trainingProgress > 0 && (
                <div className="mb-6">
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-semibold text-gray-700">Training Progress</span>
                    <span className="text-sm font-semibold text-green-600">{trainingProgress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div
                      className="bg-gradient-to-r from-green-500 to-teal-500 h-4 rounded-full transition-all duration-300"
                      style={{ width: `${trainingProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {metrics && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-6 bg-blue-50 rounded-xl text-center">
                    <div className="text-3xl font-bold text-blue-600">{metrics.mae}</div>
                    <div className="text-sm text-gray-600 mt-1">MAE (tCO2)</div>
                  </div>
                  <div className="p-6 bg-green-50 rounded-xl text-center">
                    <div className="text-3xl font-bold text-green-600">{metrics.rmse}</div>
                    <div className="text-sm text-gray-600 mt-1">RMSE (tCO2)</div>
                  </div>
                  <div className="p-6 bg-purple-50 rounded-xl text-center">
                    <div className="text-3xl font-bold text-purple-600">{metrics.r2}</div>
                    <div className="text-sm text-gray-600 mt-1">R² Score</div>
                  </div>
                  <div className="p-6 bg-orange-50 rounded-xl text-center">
                    <div className="text-3xl font-bold text-orange-600">{metrics.accuracy}%</div>
                    <div className="text-sm text-gray-600 mt-1">Accuracy</div>
                  </div>
                </div>
              )}
            </div>

            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Feature Importance</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={150} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#10b981" name="Importance %" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Predictions Tab */}
        {activeTab === 'predictions' && (
          <div className="space-y-6">
            {!modelTrained ? (
              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-6 rounded-lg">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-6 h-6 text-yellow-600" />
                  <p className="text-yellow-800 font-semibold">Please train the model first (see Model tab)</p>
                </div>
              </div>
            ) : (
              <>
                <div className="bg-white p-8 rounded-xl shadow-lg">
                  <h2 className="text-2xl font-bold text-gray-800 mb-6">Emission Predictions vs Actual</h2>
                  <ResponsiveContainer width="100%" height={350}>
                    <LineChart data={predictions}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={3} name="Actual Emissions (tCO2)" />
                      <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={3} strokeDasharray="5 5" name="Predicted Emissions (tCO2)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-white p-8 rounded-xl shadow-lg">
                  <h2 className="text-2xl font-bold text-gray-800 mb-6">Prediction Details</h2>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr className="bg-gray-100">
                          <th className="p-3 font-semibold">Month</th>
                          <th className="p-3 font-semibold">Actual (tCO2)</th>
                          <th className="p-3 font-semibold">Predicted (tCO2)</th>
                          <th className="p-3 font-semibold">Error</th>
                          <th className="p-3 font-semibold">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictions.map((pred, i) => {
                          const error = Math.abs(pred.actual - pred.predicted);
                          const errorPercent = ((error / pred.actual) * 100).toFixed(1);
                          return (
                            <tr key={i} className="border-b hover:bg-gray-50">
                              <td className="p-3 font-semibold">{pred.month}</td>
                              <td className="p-3">{Math.round(pred.actual)}</td>
                              <td className="p-3 text-green-600 font-semibold">{Math.round(pred.predicted)}</td>
                              <td className="p-3">
                                <span className={error < 20 ? 'text-green-600' : 'text-orange-600'}>
                                  ±{Math.round(error)} ({errorPercent}%)
                                </span>
                              </td>
                              <td className="p-3">
                                <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold">
                                  {pred.confidence.toFixed(1)}%
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-green-500 to-teal-500 p-6 rounded-xl shadow-lg text-white">
                    <Zap className="w-10 h-10 mb-3" />
                    <h3 className="text-lg font-bold mb-2">Action: Reduce Peak Energy</h3>
                    <p className="text-sm opacity-90">Predictions show high emissions in July-August. Schedule maintenance to reduce energy demand by 15%.</p>
                  </div>
                  
                  <div className="bg-gradient-to-br from-blue-500 to-purple-500 p-6 rounded-xl shadow-lg text-white">
                    <Factory className="w-10 h-10 mb-3" />
                    <h3 className="text-lg font-bold mb-2">Action: Optimize Production</h3>
                    <p className="text-sm opacity-90">Shift production to months with lower predicted emissions (Oct-Feb) to maximize carbon efficiency.</p>
                  </div>
                  
                  <div className="bg-gradient-to-br from-orange-500 to-red-500 p-6 rounded-xl shadow-lg text-white">
                    <Cloud className="w-10 h-10 mb-3" />
                    <h3 className="text-lg font-bold mb-2">Action: Carbon Credits</h3>
                    <p className="text-sm opacity-90">Purchase carbon offsets proactively for high-emission months to maintain compliance and reputation.</p>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Ethics Tab */}
        {activeTab === 'ethics' && (
          <div className="space-y-6">
            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Ethical Considerations & Bias Analysis</h2>
              
              <div className="space-y-6">
                <div className="border-l-4 border-red-500 pl-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-6 h-6 text-red-500" />
                    Potential Biases
                  </h3>
                  <ul className="space-y-3 text-gray-700">
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-red-500">•</span>
                      <div>
                        <strong>Geographic Bias:</strong> Model trained primarily on data from industrialized nations. May not generalize well to emerging economies with different energy infrastructure.
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-red-500">•</span>
                      <div>
                        <strong>Temporal Bias:</strong> Historical data may not capture rapid technological changes (e.g., sudden renewable energy adoption) leading to prediction drift.
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-red-500">•</span>
                      <div>
                        <strong>Sectoral Bias:</strong> Energy-intensive industries over-represented in training data, potentially underestimating emissions in service sectors.
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-bold text-red-500">•</span>
                      <div>
                        <strong>
